# Frame interpolation is used just for nicer visualization when showing interpolated human poses, so
# the background matches the poses.
# (3D human animations don't look good at 24-30 fps)
#
# This uses RIFE v4.18. The original code is available at
# https://github.com/hzwer/Practical-RIFE (MIT License).
# Publication:
# @inproceedings{huang2022rife,
#   title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
#   author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
#   booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
#   year={2022}
# }

import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7 + 16, c=192)
        self.block1 = IFBlock(8 + 4 + 16, c=128)
        self.block2 = IFBlock(8 + 4 + 16, c=96)
        self.block3 = IFBlock(8 + 4 + 16, c=64)
        self.encode = Head()

    def load_from_file(self):
        param = torch.load(osp.join(osp.dirname(__file__), 'flownet.pkl'), weights_only=True)
        modified_param = {k.replace("module.", ""): v for k, v in param.items() if "module." in k}
        self.load_state_dict(modified_param, strict=False)

    def forward(self, img0, img1, timestep=0.5, scale=1.0):
        scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]

        if not torch.is_tensor(timestep):
            timestep = torch.full_like(img0[:, :1], timestep)
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        f0 = self.encode(img0)
        f1 = self.encode(img1)
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        block = [self.block0, self.block1, self.block2, self.block3]
        for i in range(4):
            if flow is None:
                flow, mask = block[i](
                    torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=scale_list[i])
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, mask = block[i](
                    torch.cat((warped_img0, warped_img1, wf0, wf1, timestep, mask), 1),
                    flow, scale=scale_list[i])
                flow += fd
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
        mask = torch.sigmoid(mask)
        return warped_img0 * mask + warped_img1 * (1 - mask)


class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[ResConv(c) for _ in range(8)])
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 6, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, flow=None, scale=1):
        x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(
                flow, scale_factor=1. / scale, mode="bilinear",
                align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        return flow, mask


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 32, 3, 2, 1)
        self.cnn1 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(32, 8, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3


class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True)
    )


def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True)
    )


backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
            1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
            1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat(
        [tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
