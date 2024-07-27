import torch
import torch.nn as nn


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3, 3), stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm2d(chann_out),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [nn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        # self.layer6 = vgg_fc_layer(7*7*512, 4096)
        # self.layer7 = vgg_fc_layer(4096, 4096)

        # Final layer
        # self.layer8 = nn.Linear(4096, n_classes)

    def forward(self, x):
        # print(type(x))
        out = self.layer1(x)
        out = self.layer2(out)
        out_3 = self.layer3(out)
        out_4 = self.layer4(out_3)
        out_5 = self.layer5(out_4)

        # out = vgg16_features.view(out.size(0), -1)
        # out = self.layer6(out)
        # out = self.layer7(out)
        # out = self.layer8(out)

        return out_3, out_4, out_5


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class decoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(decoder, self).__init__()
        self.deconv1 = up_conv(ch_in=in_channels, ch_out=128)
        self.deconv2 = up_conv(ch_in=128, ch_out=64)
        self.deconv3 = up_conv(ch_in=64, ch_out=32)
        self.conv = BasicConv(in_planes=32, out_planes=mid_channels, kernel_size=3, padding=1)
        self.out_conv = BasicConv(in_planes=mid_channels, out_planes=out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.deconv1(x)
        x2 = self.deconv2(x1)
        x3 = self.deconv3(x2)
        x_mid = self.conv(x3)
        out = self.out_conv(x_mid)

        return x_mid, out


class fusion_net(nn.Module):
    def __init__(self):
        super(fusion_net, self).__init__()
        self.VGG16_l = VGG16()
        self.VGG16_r = VGG16()
        self.RFB3_l = BasicRFB(in_planes=256, out_planes=256)
        self.RFB4_l = BasicRFB(in_planes=512, out_planes=512)
        self.RFB5_l = BasicRFB(in_planes=512, out_planes=512)
        self.RFB3_r = BasicRFB(in_planes=256, out_planes=256)
        self.RFB4_r = BasicRFB(in_planes=512, out_planes=512)
        self.RFB5_r = BasicRFB(in_planes=512, out_planes=512)
        self.Attention5_l = Attention_block(F_g=512, F_l=512, F_int=512)
        self.Attention5_r = Attention_block(F_g=512, F_l=512, F_int=512)
        self.Attention4_l = Attention_block(F_g=512, F_l=256, F_int=256)
        self.Attention4_r = Attention_block(F_g=512, F_l=256, F_int=256)
        self.conv_34 = BasicConv(in_planes=1024, out_planes=512, kernel_size=3, padding=1, relu=False)
        self.conv_23 = BasicConv(in_planes=512, out_planes=128, kernel_size=3, padding=1)
        self.up5 = up_conv(ch_in=512, ch_out=512)
        self.up4 = up_conv(ch_in=512, ch_out=512)
        self.decoder = decoder(in_channels=256, mid_channels=3, out_channels=1)
        self.sigmoid = nn.Sigmoid()



    def forward(self, rgbl, rgbr):
        conv3_out_l, conv4_out_l, conv5_out_l = self.VGG16_l(rgbl)
        # print(conv3_out_l.shape, conv4_out_l.shape, conv5_out_l.shape)
        conv3_out_r, conv4_out_r, conv5_out_r = self.VGG16_r(rgbr)
        RFB3_out_l = self.RFB3_l(conv3_out_l)
        # print(RFB3_out_l.shape)
        RFB4_out_l = self.RFB4_l(conv4_out_l)
        # print(RFB4_out_l.shape)
        RFB5_out_l = self.RFB5_l(conv5_out_l)
        RFB3_out_r = self.RFB3_r(conv3_out_r)
        RFB4_out_r = self.RFB4_r(conv4_out_r)
        RFB5_out_r = self.RFB5_r(conv5_out_r)
        att5r_out = self.Attention5_r(self.up5(RFB5_out_l), RFB4_out_r)
        att5l_out = self.Attention5_l(self.up5(RFB5_out_r), RFB4_out_l)
        att5r_out = self.conv_34(torch.cat((RFB4_out_r, att5r_out), dim=1))
        att5l_out = self.conv_34(torch.cat((RFB4_out_l, att5l_out), dim=1))
        att4r_out = self.Attention4_r(self.up4(att5l_out), RFB3_out_r)
        att4l_out = self.Attention4_l(self.up4(att5r_out), RFB3_out_l)
        att4r_out = self.conv_23(torch.cat((RFB3_out_r, att4r_out), dim=1))
        att4l_out = self.conv_23(torch.cat((RFB3_out_l, att4l_out), dim=1))
        decoder_input = torch.cat((att4l_out, att4r_out), dim=1)
        x_fusion, out = self.decoder(decoder_input)
        out = self.sigmoid(out)

        return x_fusion, out


class rgbD_net(nn.Module):
    def __init__(self):
        super(rgbD_net, self).__init__()
        self.conv_depth = BasicConv(in_planes=1, out_planes=3, kernel_size=3, stride=1, padding=1)
        self.fusion_model = fusion_net()

    def forward(self, rgbl, rgbr):
        feature_fusion, pre_FISM = self.fusion_model(rgbl, rgbr)
        return feature_fusion, pre_FISM













