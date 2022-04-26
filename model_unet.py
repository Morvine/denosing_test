from __future__ import print_function, division
import torch.nn as nn
import torch.utils.data
import torch


class conv_block(nn.Module):

    def __init__(self, in_ch, out_ch, ks = 3, pad = 1):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=1, padding=pad, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=ks, stride=1, padding=pad, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):

    def __init__(self, in_ch, out_ch, ks = 3, pad = 1):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):

    def __init__(self, in_ch=1, out_ch=1):
        super(UNet, self).__init__()

        n1 = 16
        filters_enc = [n1, n1*2, n1 *4, n1 * 8, n1 * 16]
        filters_dec = [n1, n1*2, n1 *4, n1 * 8, n1 * 16]


        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters_enc[0])
        self.Conv2 = conv_block(filters_enc[0], filters_enc[1])
        self.Conv3 = conv_block(filters_enc[1], filters_enc[2])
        self.Conv4 = conv_block(filters_enc[2], filters_enc[3])
        self.Conv5 = conv_block(filters_enc[3], filters_enc[4])

        self.Up5 = up_conv(filters_dec[4], filters_dec[3])
        self.Up_conv5 = conv_block(filters_dec[3] + filters_enc[3], filters_dec[3])

        self.Up4 = up_conv(filters_dec[3], filters_dec[2])
        self.Up_conv4 = conv_block(filters_dec[2] + filters_enc[2], filters_dec[2])

        self.Up3 = up_conv(filters_dec[2], filters_dec[1])
        self.Up_conv3 = conv_block(filters_dec[1] + filters_enc[1], filters_dec[1])

        self.Up2 = up_conv(filters_dec[1], filters_dec[0])
        self.Up_conv2 = conv_block(filters_dec[0]+filters_dec[0], filters_dec[0])

        self.Conv = nn.Conv2d(filters_dec[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.do1 = nn.Dropout(0.15)
        self.do2 = nn.Dropout(0.15)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e4 = self.do1(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        e5 = self.do2(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        cls = self.avgpool(e5)
        cls = torch.flatten(cls, 1)
        cls = self.fc(cls)


        return out, cls

