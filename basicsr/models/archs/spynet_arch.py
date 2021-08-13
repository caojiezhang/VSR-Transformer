import math
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.models.archs.arch_util import flow_warp


class BasicModule(nn.Module):
    """Basic Module for SpyNet.
    """
    def __init__(self):
        super(BasicModule, self).__init__()

        self.basic_module = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), 
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1,padding=3), 
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), 
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), 
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))
        
    def forward(self, tensor_input):
        return self.basic_module(tensor_input)


class SpyNet(nn.Module):
    """SpyNet architecture.

    Args:
        load_path (str): path for pretrained SpyNet. Default: None.
    """

    def __init__(self, load_path=None):
        super(SpyNet, self).__init__()
        self.basic_module = nn.ModuleList([BasicModule() for _ in range(6)]) 
        if load_path:
            self.load_state_dict(
                torch.load(
                    load_path,
                    map_location=lambda storage, loc: storage)['params'])

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        tensor_output = (tensor_input - self.mean) / self.std
        return tensor_output

    def process(self, ref, supp):
        flow = []

        ref = [self.preprocess(ref)]
        supp = [self.preprocess(supp)]
        # ref = [ref]
        # supp = [supp]

        for level in range(5):
            ref.insert(
                0,
                F.avg_pool2d(
                    input=ref[0],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))
            supp.insert(
                0,
                F.avg_pool2d(
                    input=supp[0],
                    kernel_size=2,
                    stride=2,
                    count_include_pad=False))

        flow = ref[0].new_zeros([
            ref[0].size(0), 2,
            int(math.floor(ref[0].size(2) / 2.0)),
            int(math.floor(ref[0].size(3) / 2.0))
        ])

        for level in range(len(ref)):
            upsampled_flow = F.interpolate(
                input=flow,
                scale_factor=2,
                mode='bilinear',
                align_corners=True) * 2.0

            if upsampled_flow.size(2) != ref[level].size(2):
                upsampled_flow = F.pad(
                    input=upsampled_flow, pad=[0, 0, 0, 1], mode='replicate')
            if upsampled_flow.size(3) != ref[level].size(3):
                upsampled_flow = F.pad(
                    input=upsampled_flow, pad=[0, 1, 0, 0], mode='replicate')

            flow = self.basic_module[level](torch.cat([
                ref[level],
                flow_warp(
                    supp[level],
                    upsampled_flow.permute(0, 2, 3, 1),
                    interp_mode='bilinear',
                    padding_mode='border'), upsampled_flow
            ], 1)) + upsampled_flow

        return flow

    def forward(self, ref, supp):
        assert ref.size() == supp.size()

        h, w = ref.size(2), ref.size(3)
        w_floor = math.floor(math.ceil(w / 32.0) * 32.0)
        h_floor = math.floor(math.ceil(h / 32.0) * 32.0)

        ref = F.interpolate(
            input=ref,
            size=(h_floor, w_floor),
            mode='bilinear',
            align_corners=False)
        supp = F.interpolate(
            input=supp,
            size=(h_floor, w_floor),
            mode='bilinear',
            align_corners=False)

        flow = F.interpolate(
            input=self.process(ref, supp),
            size=(h, w),
            mode='bilinear',
            align_corners=False)

        flow[:, 0, :, :] *= float(w) / float(w_floor)
        flow[:, 1, :, :] *= float(h) / float(h_floor)

        return flow


"""
The following is modifid from "https://github.com/Feynman1999/basicVSR_mge"


    def upsample(self, forward_hidden, backward_hidden):
        out = self.conv4(F.concat([forward_hidden, backward_hidden], axis=1))
        out = self.reconstruction(out)
        out = self.lrelu(self.upsample1(out))
        out = self.lrelu(self.upsample2(out))
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        return out # [B, 3, 4*H, 4*W]

        b, t, c, h, w = x.size()
        forward_hiddens = []
        backward_hiddens = []
        res = []
        for i in range(t):
            x_cur = torch.cat((x[:, i, ...], x[:, t-i-1, ...]), dim=0)
            if i == 0:
                flow = self.flownet(x_cur, x_cur)
            else:
                x_ref = torch.cat((x[:, i-1, ...], x[:, t-i, ...]), dim=0)
                flow = self.flownet(x_cur, x_ref)
            hidden = backwarp(flow, x_cur) # [B, C, H, W]
            forward_hiddens.append(hidden[0:b, ...])
            backward_hiddens.append(hidden[b:2*b, ...])
        for i in range(t):
            res.append(self.upsample(forward_hiddens[i], backward_hiddens[t-i-1]))
        res = torch.stack(res, dim=1) # [B,T,3,H,W]

backwarp_tenGrid = {}
def backwarp(input, flow):
    # https://github.com/Feynman1999/basicVSR_mge
    _, _, H, W = flow.shape
    if str(flow.shape) not in backwarp_tenGrid.keys():
        x_list = np.linspace(0., W - 1., W).reshape(1, 1, 1, W)
        x_list = x_list.repeat(H, axis=2)
        y_list = np.linspace(0., H - 1., H).reshape(1, 1, H, 1)
        y_list = y_list.repeat(W, axis=3)
        xy_list = np.concatenate((x_list, y_list), 1)  # [1,2,H,W]
        backwarp_tenGrid[str(flow.shape)] = torch.tensor(xy_list.astype(np.float32))
    return F.grid_sample(input=input, grid=(backwarp_tenGrid[str(flow.shape)] + flow).permute(0, 2, 3, 1), \
            mode='bilinear', padding_mode='border')


class Basic(nn.Module):
    def __init__(self, intLevel):
        super(Basic, self).__init__()
        self.netBasic = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3), # 8=3+3+2
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
        )
    def forward(self, x):
        return self.netBasic(x)


class Spynet(nn.Module):
    def __init__(self, num_layers, pretrain_ckpt_path=None):
        super(Spynet, self).__init__()
        assert num_layers in (1, 2, 3, 4, 5)
        self.num_layers = num_layers
        self.threshold = 8
        self.pretrain_ckpt_path = pretrain_ckpt_path

        basic_list = [Basic(intLevel) for intLevel in range(num_layers)]
        self.netBasic = nn.Sequential(*basic_list)

    def preprocess(self, tenInput):
        tenRed = (tenInput[:, 0:1, :, :] - 0.485) / 0.229
        tenGreen = (tenInput[:, 1:2, :, :] - 0.456) / 0.224
        tenBlue = (tenInput[:, 2:3, :, :] - 0.406 ) / 0.225
        return torch.cat((tenRed, tenGreen, tenBlue), dim=1) # [B,3,H,W]

    def forward(self, x1, x2):
        x1 = [self.preprocess(x1)]
        x2 = [self.preprocess(x2)]

        for intLevel in range(self.num_layers - 1):
            if x1[0].shape[2] >= self.threshold or x1[0].shape[3] >= self.threshold:
                x1.insert(0, F.avg_pool2d(inp=x1[0], kernel_size=2, stride=2))
                x2.insert(0, F.avg_pool2d(inp=x2[0], kernel_size=2, stride=2))
        
        flow = torch.zeros([x1[0].shape[0], 2, int(math.floor(x1[0].shape[2] / 2.0)), int(math.floor(x1[0].shape[3] / 2.0))])
        for intLevel in range(len(x1)): 
            # normal:  5 for training  (4*4, 8*8, 16*16, 32*32, 64*64)  5 for test  (11*20, 22*40, 45*80, 90*160, 180*320)
            # small:   3 for training  (16*16, 32*32, 64*64)       3 for test  (45*80, 90*160, 180*320)
            up_flow = F.interpolate(flow, scale_factor=2, mode='BILINEAR', align_corners=True) * 2.0
            flow = self.netBasic[intLevel](torch.cat((x1[intLevel], backwarp(x2[intLevel], up_flow), up_flow), dim=1)) + up_flow
        return flow

    def init_weights(self, strict=True):
        # load ckpt from path
        if self.pretrain_ckpt_path is not None:
            print("loading pretrained model for Spynet ...")
            state_dict = torch.load(self.pretrain_ckpt_path)
            self.load_state_dict(state_dict, strict=strict)
        else:
            pass
"""