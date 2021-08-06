#!/usr/bin/python3
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import dataset


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # 权值初始化，kaiming正态分布：此为0均值的正态分布，N～ (0,std)，其中std = sqrt(2/(1+a^2)*fan_in)
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        else:
            m.initialize()


# resnet组件
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3 * dilation - 1) // 2,
                               bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(out + residual, inplace=True)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(64, 3, stride=1, dilation=1)
        self.layer2 = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3 = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4 = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * 4))

        layers = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes * 4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('./resnet50-19c8e357.pth'), strict=False)


""" Channel Attention Module """
class CALayer(nn.Module):
    def __init__(self, in_ch_left, in_ch_down):
        super(CALayer, self).__init__()
        self.conv0 = nn.Conv2d(in_ch_left, 256, kernel_size=1, stride=1, padding=0)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_ch_down, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256
        down = down.mean(dim=(2, 3), keepdim=True)
        down = F.relu(self.conv1(down), inplace=True)
        down = torch.sigmoid(self.conv2(down))
        return left * down

    def initialize(self):
        weight_init(self)


""" Body_Aggregation1 Module """
class BA1(nn.Module):
    def __init__(self, in_ch_left, in_ch_down, in_ch_right):
        super(BA1, self).__init__()
        self.conv0 = nn.Conv2d(in_ch_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_ch_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_ch_right, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv_d1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256 * 3, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True)  # 256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True)  # 256

        down_1 = self.conv_d1(down)

        w1 = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_1 * left, inplace=True)

        # z3
        down_2 = self.conv_d2(right)
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode='bilinear')
        z3 = F.relu(down_2 * left, inplace=True)

        out = torch.cat((z1, z2, z3), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

    def initialize(self):
        weight_init(self)


""" Body_Aggregation2 Module """
class BA2(nn.Module):
    def __init__(self, in_ch):
        super(BA2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]

        return F.relu(w * out1 + b, inplace=True)

    def initialize(self):
        weight_init(self)


class ConvBn(nn.Sequential):
    """
    Cascade of 2D convolution and batch norm.
    """

    def __init__(self, in_ch, out_ch, kernel_size, padding=0, dilation=1):
        super(ConvBn, self).__init__()
        self.add_module(
            "conv",
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False),
        )
        self.add_module("bn", nn.BatchNorm2d(out_ch, eps=1e-5, momentum=0.999))

    def initialize(self):
        weight_init(self)


class ASPPPooling(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBn(in_ch, out_ch, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        h = F.adaptive_avg_pool2d(x, 1)
        h = F.relu(self.conv(h))
        h = F.interpolate(h, size=(H, W), mode="bilinear", align_corners=False)
        return h

    def initialize(self):
        weight_init(self)


class ASPP(nn.Module):
    def __init__(self, in_ch, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        self.conv1 = ConvBn(in_ch, out_channels, 1)
        self.conv_aspp1 = ConvBn(in_ch, out_channels, 3, atrous_rates[0], atrous_rates[0])
        self.conv_aspp2 = ConvBn(in_ch, out_channels, 3, atrous_rates[1], atrous_rates[1])
        self.conv_aspp3 = ConvBn(in_ch, out_channels, 3, atrous_rates[2], atrous_rates[2])
        self.conv_pool = ASPPPooling(in_ch, out_channels)
        self.conv2 = ConvBn(5 * out_channels, in_ch, 1)

    def forward(self, x):
        res = []
        res.append(F.relu(self.conv1(x)))
        res.append(F.relu(self.conv_aspp1(x)))
        res.append(F.relu(self.conv_aspp2(x)))
        res.append(F.relu(self.conv_aspp3(x)))
        res.append(F.relu(self.conv_pool(x)))
        out = torch.cat([a for a in res], dim=1)
        out = F.relu(self.conv2(out))

        out = F.dropout(out, p=0.5, training=self.training)
        return out

    def initialize(self):
        weight_init(self)

class Edge_Net(nn.Module):
    def __init__(self, in_ch_list):
        super(Edge_Net, self).__init__()

        self.conv5_1 = nn.Conv2d(in_ch_list[3], 256, 1)
        self.aspp = ASPP(256, [1, 2, 4])
        self.conv5_2 = nn.Conv2d(256, 64, 1)
        self.conv5_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(64, 256, 1)

        self.conv4_1 = nn.Conv2d(in_ch_list[2], 256, 1)

        self.conv45_1 = nn.Conv2d(256, 64, 1)
        self.conv45_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv45_3 = nn.Conv2d(64, 256, 1)
        self.conv45_4 = nn.Conv2d(256, 256, 1)

        self.conv3_1 = nn.Conv2d(in_ch_list[1], 256, 1)

        self.conv345_1 = nn.Conv2d(256, 64, 1)
        self.conv345_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv345_3 = nn.Conv2d(64, 256, 1)
        self.conv345_4 = nn.Conv2d(256, 256, 1)

        self.conv345_123_1 = nn.Conv2d(256, 64, 1)
        self.conv345_123_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv345_123_3 = nn.Conv2d(64, 256, 1)
        self.conv345_123_4 = nn.Conv2d(256, 256, 1)

        self.conv2_1 = nn.Conv2d(in_ch_list[0], 64, 1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(64, 256, 1)
        self.linear256 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, out2, out3, out4, out5):
        out5_1 = F.relu(self.conv5_1(out5), inplace=False)
        out5_1 = self.aspp(out5_1)
        out5_2 = F.relu(self.conv5_2(out5_1), inplace=False)
        out5_3 = F.relu(self.conv5_3(out5_2), inplace=False)
        out5_4 = self.conv5_4(out5_3)

        out5_4_i = F.interpolate(out5_4, size=out4.size()[2:], mode='bilinear', align_corners=False)
        out4_1 = self.conv4_1(out4)

        out45 = F.relu((out5_4_i + out4_1), inplace=False)
        out45_1 = F.relu(self.conv45_1(out45), inplace=False)
        out45_2 = F.relu(self.conv45_2(out45_1), inplace=False)
        out45_3 = self.conv45_3(out45_2)
        out45_4 = self.conv45_4(out45)

        out45_3_i = F.interpolate(out45_3, size=out3.size()[2:], mode='bilinear', align_corners=False)
        out45_4_i = F.interpolate(out45_4, size=out3.size()[2:], mode='bilinear', align_corners=False)
        out3_1 = self.conv3_1(out3)

        out345 = F.relu((out45_3_i + out45_4_i + out3_1), inplace=False)
        out345_1 = F.relu(self.conv345_1(out345), inplace=False)
        out345_2 = F.relu(self.conv345_2(out345_1), inplace=False)
        out345_3 = self.conv345_3(out345_2)
        out345_4 = self.conv345_4(out345)

        out345_123 = F.relu((out345_3 + out345_4), inplace=False)
        out345_123_1 = F.relu(self.conv345_123_1(out345_123), inplace=False)
        out345_123_2 = F.relu(self.conv345_123_2(out345_123_1), inplace=False)
        out345_123_3 = self.conv345_123_3(out345_123_2)
        out345_123_4 = self.conv345_123_4(out345_123)

        out345_123_3_i = F.interpolate(out345_123_3, size=out2.size()[2:], mode='bilinear', align_corners=False)
        out345_123_4_i = F.interpolate(out345_123_4, size=out2.size()[2:], mode='bilinear', align_corners=False)
        out345_4_i = F.interpolate(out345_4, size=out2.size()[2:], mode='bilinear', align_corners=False)

        out2_1 = F.relu(self.conv2_1(out2), inplace=False)
        out2_2 = F.relu(self.conv2_2(out2_1), inplace=False)
        out2_3 = self.conv2_3(out2_2)

        out2 = F.relu((out345_123_3_i + out345_123_4_i + out345_4_i + out2_3), inplace=False)
        out2 = self.linear256(out2)
        out2_coarse = F.interpolate(out2, size=x.size()[2:], mode='bilinear')
        return out2_coarse  # ,out345,out45,out5_1

    def initialize(self):
        weight_init(self)


class GFINet(nn.Module):
    def __init__(self, cfg):
        super(GFINet, self).__init__()
        self.aspp = ASPP(256, [1, 2, 4])
        self.cfg = cfg
        self.bkbone = ResNet()

        self.ca45 = CALayer(2048, 2048)
        self.ca35 = CALayer(2048, 2048)
        self.ca25 = CALayer(2048, 2048)
        self.ca55 = CALayer(256, 2048)

        self.ba1_45 = BA1(1024, 256, 256)
        self.ba1_34 = BA1(512, 256, 256)
        self.ba1_23 = BA1(256, 256, 256)

        self.ba2_5 = BA2(256)
        self.ba2_4 = BA2(256)
        self.ba2_3 = BA2(256)
        self.ba2_2 = BA2(256)

        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        self.edge_net = Edge_Net([256, 512, 1024, 2048])
        self.conv_f = nn.Conv2d(3, 1, 1)
        self.conv_s = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.conv_t = nn.Conv2d(1, 1, 1)

        self.conv_o = nn.Conv2d(2048, 256, 1)
        self.conv_p = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_q = nn.Conv2d(256, 2048, 1)
        self.bn = nn.BatchNorm2d(1)
        self.bn256 = nn.BatchNorm2d(256)
        self.bn2048 = nn.BatchNorm2d(2048)
        self.initialize()

    def forward(self, x):
        out1, out2, out3, out4, out5_ = self.bkbone(x)

        # edge
        out2_coarse = self.edge_net(x, out2, out3, out4, out5_)

        out5_f = F.relu(self.conv_o(out5_))  # 256 1*1
        out5_fp = self.aspp(out5_f)  # 256
        out5_f = self.bn2048(self.conv_q(out5_fp))  # 1*1

        out5_f = F.sigmoid(out5_f)
        out5_ = out5_ + out5_f * out5_

        # CA
        out4_a = self.ca45(out5_, out5_)
        out3_a = self.ca35(out5_, out5_)
        out2_a = self.ca25(out5_, out5_)

        out5 = out5_fp

        out4 = self.ba2_4(self.ba1_45(out4, out5, out4_a))
        out3 = self.ba2_3(self.ba1_34(out3, out4, out3_a))
        out2 = self.ba2_2(self.ba1_23(out2, out3, out2_a))

        out5 = self.linear5(out5)
        out4 = self.linear4(out4)
        out3 = self.linear3(out3)
        out2 = self.linear2(out2)

        out5 = F.interpolate(out5, size=x.size()[2:], mode='bilinear')
        out4 = F.interpolate(out4, size=x.size()[2:], mode='bilinear')
        out3 = F.interpolate(out3, size=x.size()[2:], mode='bilinear')
        out2 = F.interpolate(out2, size=x.size()[2:], mode='bilinear')


        out2_f = torch.cat((out2, out2_coarse * F.sigmoid(out2), (1 - F.sigmoid(out2_coarse)) * out2), dim=1)
        out2_f = F.relu(self.bn(self.conv_f(out2_f)))
        out2_f = self.bn(self.conv_s(out2_f))
        out2_f = F.sigmoid(out2_f)
        out2 = out2 + out2 * out2_f
        out2 = self.conv_t(out2)

        out3_f = torch.cat((out3, out2_coarse * F.sigmoid(out3), (1 - F.sigmoid(out2_coarse)) * out3), dim=1)
        out3_f = F.relu(self.bn(self.conv_f(out3_f)))
        out3_f = self.bn(self.conv_s(out3_f))
        out3_f = F.sigmoid(out3_f)
        out3 = out3 + out3 * out3_f
        out3 = self.conv_t(out3)

        out4_f = torch.cat((out4, out2_coarse * F.sigmoid(out4), (1 - F.sigmoid(out2_coarse)) * out4), dim=1)
        out4_f = F.relu(self.bn(self.conv_f(out4_f)))
        out4_f = self.bn(self.conv_s(out4_f))
        out4_f = F.sigmoid(out4_f)
        out4 = out4 + out4 * out4_f
        out4 = self.conv_t(out4)

        out5_f = torch.cat((out5, out2_coarse * F.sigmoid(out5), (1 - F.sigmoid(out2_coarse)) * out5), dim=1)
        out5_f = F.relu(self.bn(self.conv_f(out5_f)))
        out5_f = self.bn(self.conv_s(out5_f))
        out5_f = F.sigmoid(out5_f)
        out5 = out5 + out5 * out5_f
        out5 = self.conv_t(out5)

        return out2, out3, out4, out5, out2_coarse  # , out3_coarse, out4_coarse, out5_coarse

    def initialize(self):
        if self.cfg.snapshot:  
            try:
                self.load_state_dict(torch.load(self.cfg.snapshot))
            except:
                print("Warning: please check the snapshot file:", self.cfg.snapshot)
                pass
        else:
            weight_init(self)

