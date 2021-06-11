import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import load_checkpoint

from mmedit.models.common import (PixelShufflePack, ResidualBlockNoBN, flow_warp, make_layer)
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


class SPyNet(nn.Module):
    """SPyNet network structure.
    The difference to the SPyNet in [tof.py] is that
        1. more SPyNetBasicModule is used in this version, and
        2. no batch normalization is used in this version.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    Args:
        pretrained (str): path for pre-trained SPyNet. Default: None.
    """

    def __init__(self, pretrained):
        super().__init__()

        self.basic_module = nn.ModuleList([SPyNetBasicModule() for _ in range(6)])

        if isinstance(pretrained, str):
            self.load_state_dict(torch.load(pretrained), strict=True)
        elif pretrained is not None:
            raise TypeError('[pretrained] should be str or None, 'f'but got {type(pretrained)}.')

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_flow(self, ref, supp):
        """Compute flow from ref to supp.
        Note that in this function, the images are already resized to a multiple of 32.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        n, _, h, w = ref.size()

        # normalize the input images
        ref = [(ref - self.mean) / self.std]
        supp = [(supp - self.mean) / self.std]

        # generate downsampled frames
        for level in range(5):
            ref.append(F.avg_pool2d(input=ref[-1], kernel_size=2, stride=2, count_include_pad=False))
            supp.append(F.avg_pool2d(input=supp[-1], kernel_size=2, stride=2, count_include_pad=False))
        ref = ref[::-1]
        supp = supp[::-1]

        # flow computation
        flow = ref[0].new_zeros(n, 2, h // 32, w // 32)
        for level in range(len(ref)):
            if level == 0:
                flow_up = flow
            else:
                flow_up = F.interpolate(input=flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            # add the residue to the upsampled flow
            flow = flow_up + self.basic_module[level](
                torch.cat([ref[level], flow_warp(supp[level], flow_up.permute(0, 2, 3, 1), padding_mode='border'), flow_up], 1))

        return flow

    def forward(self, ref, supp):
        """Forward function of SPyNet.
        This function computes the optical flow from ref to supp.
        Args:
            ref (Tensor): Reference image with shape of (n, 3, h, w).
            supp (Tensor): Supporting image with shape of (n, 3, h, w).
        Returns:
            Tensor: Estimated optical flow: (n, 2, h, w).
        """
        # upsize to a multiple of 32
        h, w = ref.shape[2:4]
        w_up = w if (w % 32) == 0 else 32 * (w // 32 + 1)
        h_up = h if (h % 32) == 0 else 32 * (h // 32 + 1)
        ref = F.interpolate(
            input=ref, size=(h_up, w_up), mode='bilinear', align_corners=False)
        supp = F.interpolate(input=supp, size=(h_up, w_up), mode='bilinear', align_corners=False)

        # compute flow, and resize back to the original resolution
        flow = F.interpolate(input=self.compute_flow(ref, supp), size=(h, w), mode='bilinear', align_corners=False)

        # adjust the flow values
        flow[:, 0, :, :] *= float(w) / float(w_up)
        flow[:, 1, :, :] *= float(h) / float(h_up)

        return flow


class SPyNetBasicModule(nn.Module):
    """Basic Module for SPyNet.
    Paper:
        Optical Flow Estimation using a Spatial Pyramid Network, CVPR, 2017
    """
    def __init__(self):
        super().__init__()

        self.basic_module = nn.Sequential(
            ConvModule(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3, norm_cfg=None, act_cfg=None))

    def forward(self, tensor_input):
        """
        Args:
            tensor_input (Tensor): Input tensor with shape (b, 8, h, w).
                8 channels contain:
                [reference image (3), neighbor image (3), initial flow (2)].
        Returns:
            Tensor: Refined flow with shape (b, 2, h, w)
        """
        return self.basic_module(tensor_input)


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(make_layer(ResidualBlockNoBN, num_blocks, mid_channels=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.
        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)
        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)