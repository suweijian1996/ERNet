import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
from utils.utils import GaussianBlur2d
from DWT_IDWT.DWT_IDWT_layer import *
class Gradient(nn.Module):
    def __init__(self, channels=2):
        super(Gradient, self).__init__()
        self.channels = channels
        kernel = [[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=1, groups=self.channels)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes=2, ratio=16):
        super(ChannelAttention, self).__init__()
        self.gradient = Gradient(channels=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Conv2d(in_planes, in_planes * ratio, 1, bias=False)
        self.lrelu1 = nn.LeakyReLU(0.2,True)
        self.fc2 = nn.Conv2d(in_planes * ratio, in_planes, 1, bias=False)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        avg_out = self.fc2(self.lrelu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.lrelu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        weight = self.softmax(out)
        #------------#
        #  2c---1c
        #------------#
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out,max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        x = self.lrelu1(x)
        #-------------#
        #mut
        #------------#
        feature = x*weight
        return self.sigmoid(feature)


# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
#         x = torch.cat([avg_out, max_out], dim=1)
#         x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
#         return self.sigmoid(x)
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:9]
    def forward(self, input):
        input = torch.cat((input,input,input),1)
        feature = self.vgg(input)
        return feature

class Downsample(nn.Module):
    def __init__(self, wavename = 'haar'):
        super(Downsample, self).__init__()
        self.dwt = DWT_2D_tiny(wavename = wavename)

    def forward(self, input):
        LL = self.dwt(input)
        return LL
class Downsample(nn.Module):
    """
        for ResNet_A
        X --> 1/4*(X_ll + X_lh + X_hl + X_hh)
    """
    def __init__(self, wavename = 'haar'):
        super(Downsample, self).__init__()
        self.dwt = DWT_2D(wavename = wavename)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        return (LL + LH + HL + HH) / 4

class DisIR_net(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(DisIR_net, self).__init__()
        # self.dwt = Downsample(wavename = 'haar')
        # self.gradient = Gradient()
        # self.gauss = GaussianBlur2d((3, 3), (2, 2))
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True)
            )

        self.C2 = nn.Sequential(
            # Downsample(wavename='haar'),
            nn.Conv2d(in_channels=128, out_channels=64,
                      kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True))
        self.C3 = nn.Sequential(
            # Downsample(wavename='haar'),
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, stride=2, padding=1,bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,True)
        )
        # self.C4 = nn.Sequential(
        #     Downsample( wavename ='haar'),
        #     nn.Conv2d(in_channels=256, out_channels=128,
        #               kernel_size=3, stride=1, padding=1,bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2,True),
        # )
        # self.C5 = nn.Sequential(
        #     Downsample(wavename='haar'),
        #     nn.Conv2d(in_channels=128, out_channels=64,
        #               kernel_size=3, stride=1, padding=1,bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2, True)
        # )
        #
        # self.F1 = nn.Sequential(
        #     Downsample(wavename='haar'),
        #     nn.Conv2d(in_channels=64, out_channels=64,
        #               kernel_size=3, stride=1, padding=1,bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2, True),
        # )
        # self.F2 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=32,
        #               kernel_size=3, stride=1, padding=1,bias=False),
        #     nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2, True)
        # )
        self.L1 = nn.Sequential(
            nn.Linear(2048,1024,bias=False),
            nn.LeakyReLU(0.2, True)
        )
        self.L2 = nn.Sequential(
            nn.Linear(1024, 1, bias=False))

    def forward(self, input):
        """Standard forward."""
        # int = torch.max(input[:,[0],:,:],input[:,[1],:,:])
        # max_out, _ = torch.max(input, dim=1, keepdim=True)
        #substract = torch.abs(input[:,[0],:,:]-input[:,[1],:,:])
        # gradient,_ = torch.max(self.gradient(input), dim=1, keepdim=True)
        #out = self.C1(torch.cat((max,substract),1))
        # input = self.att(input)*input

        out = self.C1(input)
        out = self.C2(out)
        out = self.C3(out)
        out = out.view(out.shape[0],-1)
        out = self.L1(out)
        out = self.L2(out)
        return out


class DisVIS_net(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(DisVIS_net, self).__init__()
        self.C1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True))
        self.C2 = nn.Sequential(
            # Downsample(wavename='haar'),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,True))
        self.C3 = nn.Sequential(
            # Downsample(wavename='haar'),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,True)
        )
        self.C4 = nn.Sequential(
            # Downsample(wavename='haar'),
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,True))
        self.C5 = nn.Sequential(
            # Downsample(wavename='haar'),
            nn.Conv2d(in_channels=512, out_channels=1,
                      kernel_size=3, stride=1, padding=1))
        # self.L1 = nn.Sequential(
        #     nn.Linear(4096,1024),
        #     nn.LeakyReLU(0.2,True))
        # self.L2 = nn.Sequential(
        #     nn.Linear(1024,1))
        # self.C6 = nn.Sequential(
        #     nn.Conv2d(in_channels=256, out_channels=1,
        #               kernel_size=3, stride=1, padding=1))

    def forward(self, input):

        """Standard forward."""
        out = self.C1(input)
        out = self.C2(out)
        out = self.C3(out)
        out = self.C4(out)
        out = self.C5(out)
        # out = out.view(out.shape[0],-1)
        # out = self.L1(out)
        # out = self.L2(out)
        # out = self.C6(out)
        return out

# # # #.permute(0, 3, 2, 1).contiguous()
# vgg = VGG().cuda()
# net = DisIR_net().cuda()
# a = torch.randn([1,256,64,64]).cuda()
# # b = vgg(a)
# b = net(a)
# print(b.shape)