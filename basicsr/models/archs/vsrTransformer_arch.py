import torch
from torch import nn as nn
from torch.nn import functional as F

import numpy as np
from basicsr.models.archs.arch_util import (ResidualBlockNoBN, make_layer, RCAB, ResidualGroup, default_conv, RCABWithInputConv)
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from positional_encodings import PositionalEncodingPermute3D
from torch.nn import init
import math
from torch import einsum
from basicsr.models.archs.spynet import SPyNet, SPyNetBasicModule, ResidualBlocksWithInputConv
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint
from mmedit.models.common import (PixelShufflePack, flow_warp)  # make_layer
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
import pdb


# Transformer
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, num_feat, feat_size, fn):
        super().__init__()
        self.norm = nn.LayerNorm([num_feat, feat_size, feat_size])
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class FeedForward(nn.Module):
    def __init__(self, num_feat):
        super().__init__()
        
        self.backward_resblocks = ResidualBlocksWithInputConv(num_feat+3, num_feat, num_blocks=30)
        self.forward_resblocks = ResidualBlocksWithInputConv(num_feat+3, num_feat, num_blocks=30)
        self.fusion = nn.Conv2d(num_feat*2, num_feat, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        
    def forward(self, x, lrs=None, flows=None):
        b, t, c, h, w = x.shape
        x1 = torch.cat([x[:, 1:, :, :, :], x[:, -1, :, :, :].unsqueeze(1)], dim=1)  # [B, 5, 64, 64, 64]
        flow1 = flows[1].contiguous().view(-1, 2, h, w).permute(0, 2, 3, 1)         # [B*5, 64, 64, 2]
        x1 = flow_warp(x1.view(-1, c, h, w), flow1)                                 # [B*5, 64, 64, 64]
        x1 = torch.cat([lrs.view(b*t, -1, h, w), x1], dim=1)                        # [B*5, 67, 64, 64]
        x1 = self.backward_resblocks(x1)                                            # [B*5, 64, 64, 64]

        x2 = torch.cat([x[:, 0, :, :, :].unsqueeze(1), x[:, :-1, :, :, :]], dim=1)  # [B, 5, 64, 64, 64]
        flow2 = flows[0].contiguous().view(-1, 2, h, w).permute(0, 2, 3, 1)         # [B*5, 64, 64, 2]
        x2 = flow_warp(x2.view(-1, c, h, w), flow2)                                 # [B*5, 64, 64, 64]
        x2 = torch.cat([lrs.view(b*t, -1, h, w), x2], dim=1)                        # [B*5, 67, 64, 64]
        x2 = self.forward_resblocks(x2)                                             # [B*5, 64, 64, 64]

        # fusion the backward and forward features
        out = torch.cat([x1, x2], dim=1)      # [B*5, 128, 64, 64]
        out = self.lrelu(self.fusion(out))    # [B*5, 64, 64, 64]
        out = out.view(x.shape)               # [B, 5, 64, 64, 64] 

        return out


class MatmulNet(nn.Module):
    def __init__(self) -> None:
        super(MatmulNet, self).__init__()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = torch.matmul(x, y)
        return x


class globalAttention(nn.Module):
    def __init__(self, num_feat=64, patch_size=8, heads=1):
        super(globalAttention, self).__init__()
        self.heads = heads
        self.dim = patch_size ** 2 * num_feat
        self.hidden_dim = self.dim // heads
        self.num_patch = (64 // patch_size) ** 2
        
        self.to_q = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat) 
        self.to_k = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1, groups=num_feat)
        self.to_v = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1)

        self.conv = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, padding=1)

        self.feat2patch = torch.nn.Unfold(kernel_size=patch_size, padding=0, stride=patch_size)
        self.patch2feat = torch.nn.Fold(output_size=(64, 64), kernel_size=patch_size, padding=0, stride=patch_size)

    def forward(self, x):
        b, t, c, h, w = x.shape                                # B, 5, 64, 64, 64
        H, D = self.heads, self.dim
        n, d = self.num_patch, self.hidden_dim

        q = self.to_q(x.view(-1, c, h, w))                     # [B*5, 64, 64, 64]    
        k = self.to_k(x.view(-1, c, h, w))                     # [B*5, 64, 64, 64]   
        v = self.to_v(x.view(-1, c, h, w))                     # [B*5, 64, 64, 64]

        unfold_q = self.feat2patch(q)                          # [B*5, 8*8*64, 8*8]
        unfold_k = self.feat2patch(k)                          # [B*5, 8*8*64, 8*8]  
        unfold_v = self.feat2patch(v)                          # [B*5, 8*8*64, 8*8] 

        unfold_q = unfold_q.view(b, t, H, d, n)                # [B, 5, H, 8*8*64/H, 8*8]
        unfold_k = unfold_k.view(b, t, H, d, n)                # [B, 5, H, 8*8*64/H, 8*8]
        unfold_v = unfold_v.view(b, t, H, d, n)                # [B, 5, H, 8*8*64/H, 8*8]

        unfold_q = unfold_q.permute(0,2,3,1,4).contiguous()    # [B, H, 8*8*64/H, 5, 8*8]
        unfold_k = unfold_k.permute(0,2,3,1,4).contiguous()    # [B, H, 8*8*64/H, 5, 8*8]
        unfold_v = unfold_v.permute(0,2,3,1,4).contiguous()    # [B, H, 8*8*64/H, 5, 8*8]

        unfold_q = unfold_q.view(b, H, d, t*n)                 # [B, H, 8*8*64/H, 5*8*8]
        unfold_k = unfold_k.view(b, H, d, t*n)                 # [B, H, 8*8*64/H, 5*8*8]
        unfold_v = unfold_v.view(b, H, d, t*n)                 # [B, H, 8*8*64/H, 5*8*8]

        attn = torch.matmul(unfold_q.transpose(2,3), unfold_k) # [B, H, 5*8*8, 5*8*8]
        attn = attn * (d ** (-0.5))                            # [B, H, 5*8*8, 5*8*8]
        attn = F.softmax(attn, dim=-1)                         # [B, H, 5*8*8, 5*8*8]

        attn_x = torch.matmul(attn, unfold_v.transpose(2,3))   # [B, H, 5*8*8, 8*8*64/H]
        attn_x = attn_x.view(b, H, t, n, d)                    # [B, H, 5, 8*8, 8*8*64/H]
        attn_x = attn_x.permute(0, 2, 1, 4, 3).contiguous()    # [B, 5, H, 8*8*64/H, 8*8]
        attn_x = attn_x.view(b*t, D, n)                        # [B*5, 8*8*64, 8*8]
        feat = self.patch2feat(attn_x)                         # [B*5, 64, 64, 64]
        
        out = self.conv(feat).view(x.shape)                    # [B, 5, 64, 64, 64]
        out += x                                               # [B, 5, 64, 64, 64]

        return out


class Transformer(nn.Module):
    def __init__(self, num_feat, feat_size, depth, patch_size, heads, kernel_size):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(num_feat, feat_size, globalAttention(num_feat, patch_size, heads, kernel_size))), 
                Residual(PreNorm(num_feat, feat_size, FeedForward(num_feat)))
            ]))
            
    def forward(self, x, lrs=None, flows=None):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x, lrs=lrs, flows=flows)
        return x


class Flow_spynet(nn.Module):
    def __init__(self, spynet_pretrained=None):
        super(Flow_spynet, self).__init__()
        # optical flow network for feature alignment
        self.spynet = SPyNet(pretrained=spynet_pretrained)
        
    def check_if_mirror_extended(self, lrs):
        """Check whether the input is a mirror-extended sequence.
        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the (t-1-i)-th frame.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        """
        self.is_mirror_extended = False
        if lrs.size(1) % 2 == 0:
            lrs_1, lrs_2 = torch.chunk(lrs, 2, dim=1)
            if torch.norm(lrs_1 - lrs_2.flip(1)) == 0:
                self.is_mirror_extended = True
    
    def forward(self, lrs):
        """Compute optical flow using SPyNet for feature warping.
        Note that if the input is an mirror-extended sequence, 'flows_forward' is not needed, since it is equal to 'flows_backward.flip(1)'.
        Args:
            lrs (tensor): Input LR images with shape (n, t, c, h, w)
        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """
        n, t, c, h, w = lrs.size()    
        assert h >= 64 and w >= 64, ('The height and width of inputs should be at least 64, 'f'but got {h} and {w}.')
        
        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lrs)

        lrs_1 = torch.cat([lrs[:, 0, :, :, :].unsqueeze(1), lrs], dim=1).reshape(-1, c, h, w)  # [b*6, 3, 64, 64]
        lrs_2 = torch.cat([lrs, lrs[:, t-1, :, :, :].unsqueeze(1)], dim=1).reshape(-1, c, h, w)  # [b*6, 3, 64, 64]
        
        flows_backward = self.spynet(lrs_1, lrs_2).view(n, t+1, 2, h, w)         # [b, 6, 2, 64, 64]
        flows_backward = flows_backward[:, 1:, :, :, :]                          # [b, 5, 2, 64, 64]

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lrs_2, lrs_1).view(n, t+1, 2, h, w)      # [b, 6, 2, 64, 64]
            flows_forward = flows_forward[:, :-1, :, :, :]                       # [b, 5, 2, 64, 64]

        return flows_forward, flows_backward


class vsrTransformer(nn.Module):
    def __init__(self,
                 image_ch=3,
                 num_feat=64,
                 feat_size=64,
                 num_frame=5, 
                 num_extract_block=5,
                 num_reconstruct_block=30,
                 depth=10,
                 heads=1,
                 patch_size=8,
                 spynet_pretrained=None
                 ):
        super(vsrTransformer, self).__init__()
        self.num_reconstruct_block = num_reconstruct_block
        self.center_frame_idx = num_frame // 2
        self.num_frame = num_frame

        # Feature extractor
        self.conv_first = nn.Conv2d(image_ch, num_feat, 3, 1, 1)
        self.feature_extraction = make_layer(ResidualBlockNoBN, num_extract_block, num_feat=num_feat)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Optical flow
        self.spynet = Flow_spynet(spynet_pretrained)

        # Transformer
        self.pos_embedding = PositionalEncodingPermute3D(num_frame)
        self.transformer = Transformer(num_feat, feat_size, depth, patch_size, heads)

        # Reconstruction
        self.reconstruction = make_layer(ResidualBlockNoBN, num_reconstruct_block, num_feat=num_feat)

        # Upsampling
        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, image_ch, 3, 1, 1)
        self.img_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        
    def forward(self, x): 
        
        b, t, c, h, w = x.size()                                         # [B, 5, 3, 64, 64]
        assert h%4==0 and w%4==0, ('w and h must be multiple of 4')

        # extract features for each frame
        feat = self.lrelu(self.conv_first(x.view(-1, c, h, w)))          # [B*5, 64, 64, 64]
        feat = self.feature_extraction(feat).view(b, t, -1, h, w)        # [B, 5, 64, 64, 64]

        # compute optical flow
        flows = self.spynet(x)                                           # [B, 4, 2, 64, 64] x 2
        
        # transformer
        feat = feat + self.pos_embedding(feat)                           # [B, 5, 64, 64, 64]
        tr_feat = self.transformer(feat, x, flows)                       # [B, 5, 64, 64, 64] 
        feat = tr_feat.view(b*t, -1, h, w)                               # [B*5, 64, 64, 64]

        # reconstruction
        feat = self.reconstruction(feat)                                 # [B, 64, 64, 64]   -> [B*5, 64, 64, 64]
        out = self.lrelu(self.pixel_shuffle(self.upconv1(feat)))         # [B, 64, 128, 128] -> [B*5, 64, 128, 128]
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))          # [B, 64, 256, 256] -> [B*5, 64, 256, 256]
        out = self.lrelu(self.conv_hr(out))                              # [B, 64, 256, 256] -> [B*5, 64, 256, 256]
        out = self.conv_last(out)                                        # [B, 3, 256, 256]  -> [B*5, 3, 256, 256]
        
        # residual
        base = self.img_upsample(x.view(-1, c, h, w))                    # [B*5, 3, 256, 256]
        out += base                                                      # [B*5, 3, 256, 256]
        out = out.view(b, t, c, h*4, w*4)                                # [B, 5, 3, 256, 256]

        return out

