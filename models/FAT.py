# code copy from: https://github.com/SZUcsh/FAT-Net
import torch
from torchvision import models as resnet_model
from torch import nn
from .model_utils import *
from .CTM_utils import *


class FAMBlock(nn.Module):
    def __init__(self, channels):
        super(FAMBlock, self).__init__()

        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x3 = self.conv3(x)
        x3 = self.relu3(x3)
        x1 = self.conv1(x)
        x1 = self.relu1(x1)
        out = x3 + x1

        return out

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels, n_filters, use_transpose=True):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        if use_transpose:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
                ),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Upsample(scale_factor=2, align_corners=True, mode="bilinear")

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y

class FAT_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super(FAT_Net, self).__init__()

        #transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.transformer = TransformerDown(in_channels=3, out_channels=192, image_size=img_size)
        # self.transformer = Transformer_sparse1(in_channels=3, out_channels=192, image_size=img_size)
        # self.transformer = Transformer_sparse2(in_channels=3, out_channels=192, image_size=img_size)

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)


    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf = self.transformer(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)

        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out
    
    def infere(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, ftokens, attmaps = self.transformer.infere(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, ftokens, attmaps

class FAT_Net_AS(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super(FAT_Net_AS, self).__init__()

        # transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.transformer = TransformerDown(in_channels=3, out_channels=192, image_size=img_size)
        self.transformer = TransformerDown_AS(in_channels=3, out_channels=192, image_size=img_size)
        # self.transformer = Transformer_sparse1(in_channels=3, out_channels=192, image_size=img_size)
        # self.transformer = Transformer_sparse2(in_channels=3, out_channels=192, image_size=img_size)

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, as_out = self.transformer(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)

        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, as_out

    def infere(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, ftokens, attmaps = self.transformer.infere(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, ftokens, attmaps

class FAT_Net_DTMFormerV2(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super(FAT_Net_DTMFormerV2, self).__init__()

        # transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.transformer = TransformerDown_DTMFormerV2(in_channels=3, out_channels=192, image_size=img_size)

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, as_out = self.transformer(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)

        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, as_out

    def infere(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, ftokens, attmaps = self.transformer.infere(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, ftokens, attmaps

class FAT_Net_DTMFormerV2_attnloss(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super(FAT_Net_DTMFormerV2_attnloss, self).__init__()

        # transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.transformer = TransformerDown_DTMFormerV2_attnloss(in_channels=3, out_channels=192, image_size=img_size)

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, as_out, attns = self.transformer(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)

        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, as_out, attns

    def infere(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, ftokens, attmaps = self.transformer.infere(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, ftokens, attmaps


class FAT_Net_DTMFormerV2_attnlossDS(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super(FAT_Net_DTMFormerV2_attnlossDS, self).__init__()

        # transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.transformer = TransformerDown_DTMFormerV2_attnlossDS(in_channels=3, out_channels=192, image_size=img_size)

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, as_out, attns, feature_DS  = self.transformer(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)

        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, as_out, attns, feature_DS

    def infere(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, ftokens, attmaps = self.transformer.infere(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, ftokens, attmaps

class FAT_Net_DTMFormerV2_attnlossDS(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super(FAT_Net_DTMFormerV2_attnlossDS, self).__init__()

        # transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.transformer = TransformerDown_DTMFormerV2_attnlossDSTS(in_channels=3, out_channels=192, image_size=img_size)

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, as_out, attns, feature_DS  = self.transformer(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)

        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, as_out, attns, feature_DS

    def infere(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, ftokens, attmaps = self.transformer.infere(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, ftokens, attmaps



class FAT_Net_DTMFormerV2_attnlossDSTS(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super(FAT_Net_DTMFormerV2_attnlossDSTS, self).__init__()

        # transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.transformer = TransformerDown_DTMFormerV2_attnlossDSTS(in_channels=3, out_channels=192, image_size=img_size)

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, as_out, attns, feature_DS, token_weights, idx_clusters  = self.transformer(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)

        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, as_out, attns, feature_DS, token_weights, idx_clusters

    def infere(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, ftokens, attmaps = self.transformer.infere(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, ftokens, attmaps


class FAT_Net_DTMFormerV2VarianceStageDict(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super(FAT_Net_DTMFormerV2VarianceStageDict, self).__init__()

        # transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.transformer = TransformerDown_DTMFormerV2VarianceStageDict(in_channels=3, out_channels=192, image_size=img_size)


        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, attnScores, attns, outs, variances = self.transformer(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)

        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, attnScores, attns, outs, variances

    def infere(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, ftokens, attmaps = self.transformer.infere(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, ftokens, attmaps


class FAT_Net_DTMFormerV2FirstStageChoose(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super(FAT_Net_DTMFormerV2FirstStageChoose, self).__init__()

        # transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.transformer = TransformerDown_DTMFormerV2FirstStageChoose(in_channels=3, out_channels=192, image_size=img_size)


        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, attnScores, attns, outs, variances, scores = self.transformer(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)

        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, attnScores, attns, outs, variances, scores

    def infere(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, ftokens, attmaps = self.transformer.infere(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, ftokens, attmaps



class FAT_Net_DTMFormerV3(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super(FAT_Net_DTMFormerV3, self).__init__()

        # transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.transformer = TransformerDown_DTMFormerV3(in_channels=3, out_channels=192, image_size=img_size)

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, as_out = self.transformer(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)

        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, as_out

    def infere(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, ftokens, attmaps = self.transformer.infere(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, ftokens, attmaps


class FAT_Net_sparse1(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super(FAT_Net_sparse1, self).__init__()

        # transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.transformer = TransformerDown(in_channels=3, out_channels=192, image_size=img_size)
        self.transformer = Transformer_sparse1(in_channels=3, out_channels=192, image_size=img_size)
        # self.transformer = Transformer_sparse2(in_channels=3, out_channels=192, image_size=img_size)

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf = self.transformer(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)

        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out

    def infere(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, ftokens, attmaps = self.transformer.infere(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, ftokens, attmaps

class FAT_Net_sparse2(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=256):
        super(FAT_Net_sparse2, self).__init__()

        # transformer = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_distilled_patch16_224', pretrained=True)
        resnet = resnet_model.resnet34(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # self.transformer = TransformerDown(in_channels=3, out_channels=192, image_size=img_size)
        # self.transformer = Transformer_sparse1(in_channels=3, out_channels=192, image_size=img_size)
        self.transformer = Transformer_sparse2(in_channels=3, out_channels=192, image_size=img_size)

        self.conv_seq_img = nn.Conv2d(in_channels=192, out_channels=512, kernel_size=1, padding=0)
        self.se = SEBlock(channel=1024)
        self.conv2d = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.FAMBlock1 = FAMBlock(channels=64)
        self.FAMBlock2 = FAMBlock(channels=128)
        self.FAMBlock3 = FAMBlock(channels=256)
        self.FAM1 = nn.ModuleList([self.FAMBlock1 for i in range(6)])
        self.FAM2 = nn.ModuleList([self.FAMBlock2 for i in range(4)])
        self.FAM3 = nn.ModuleList([self.FAMBlock3 for i in range(2)])

        filters = [64, 128, 256, 512]
        self.decoder4 = DecoderBottleneckLayer(filters[3], filters[2])
        self.decoder3 = DecoderBottleneckLayer(filters[2], filters[1])
        self.decoder2 = DecoderBottleneckLayer(filters[1], filters[0])
        self.decoder1 = DecoderBottleneckLayer(filters[0], filters[0])

        self.final_conv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.final_relu1 = nn.ReLU(inplace=True)
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nn.ReLU(inplace=True)
        self.final_conv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf = self.transformer(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)

        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out

    def infere(self, x):
        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)

        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        feature_cnn = self.encoder4(e3)

        feature_tf, ftokens, attmaps = self.transformer.infere(x)

        feature_tf = self.conv_seq_img(feature_tf)

        feature_cat = torch.cat((feature_cnn, feature_tf), dim=1)
        feature_att = self.se(feature_cat)
        feature_out = self.conv2d(feature_att)

        for i in range(2):
            e3 = self.FAM3[i](e3)
        for i in range(4):
            e2 = self.FAM2[i](e2)
        for i in range(6):
            e1 = self.FAM1[i](e1)
        d4 = self.decoder4(feature_out) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1

        out1 = self.final_conv1(d2)
        out1 = self.final_relu1(out1)
        out = self.final_conv2(out1)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out, ftokens, attmaps

 # ==========================================  some modules for the anti oversmoothing methods ====================================
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .components.anti_over_smoothing import Transformer_Vanilla, Transformer_Refiner, Transformer_Layerscale, Transformer_Reattention

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class TransformerDown(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, depth=12, dmodel=1024, mlp_dim=2048, patch_size=16, heads=12, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel*4

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_Vanilla(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout, num_patches)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height//patch_height),
        )

    def forward(self, x):
        # torch.Size([4, 3, 256, 256])
        x = self.to_patch_embedding(x)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        # torch.Size([4, 256, 192])
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        # transformer layer
        ax = self.transformer(x)
        # torch.Size([4, 256, 192])
        out = self.recover_patch_embedding(ax)
        # torch.Size([4, 192, 16, 16])

        return out
    
    def infere(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        xin = self.dropout(x)
        # encoder
        ax, ftokens, attmaps = self.transformer.infere(xin)  # b h*w ppc
        ftokens.insert(0, xin)
        out = self.recover_patch_embedding(ax)
        return out, ftokens, attmaps


class TransformerDown_AS(nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=192, total_depth=12,
            patch_size=16, embed_dims=[96, 192], num_heads=[4, 8], mlp_ratios=[4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2], as_depth=[1, 1],
            sr_ratios=[8, 4], num_stages=4, pretrained=None,
            k=5, sample_ratios=0.125, classes=4,
            # k=5, sample_ratios=0.0625,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.as_depth = as_depth
        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size/patch_size)


        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )

        self.recover = MTA_FAT_light_smallModule()

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([Block_AS(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(as_depth[0])])
        self.norm0_as = norm_layer(embed_dims[0])
        self.ctm_as = CTM_as(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1_as = nn.ModuleList([TCBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(as_depth[1])])
        self.norm1_as = norm_layer(embed_dims[1])

        self.stage0 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for j in range(depths[0])])
        self.norm0 = norm_layer(embed_dims[0])
        self.ctm1 = CTM_as(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([TCBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(depths[1])])
        self.norm1 = norm_layer(embed_dims[1])



    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        H = self.patch_num
        W = self.patch_num

        for blk in self.stage0_as:
            x, attn = blk(x, H, W)
        x = self.norm0_as(x)
        self.cur += self.as_depth[0]

        # attn :torch.Size([4, 4, 1024, 1024])    attn[:,0:2] :torch.Size([4, 2, 1024, 1024])
        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)  # attn_map :torch.Size([4, 1024])

        # 归一化方法1：sigmoid-0.5 (0-0.5)
        # as_out = torch.sigmoid(attn_map) - 0.5
        # 归一化方法2：(sigmoid-0.5)*2 (0-1)
        # as_out = (torch.sigmoid(attn_map) - 0.5) * 2
        # 归一化方法3：均匀归一化 (0-1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        as_out = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)

        as_out = rearrange(as_out, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm_as(token_dict, as_out, ctm_stage=1)

        for j, blk in enumerate(self.stage1_as):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1_as(token_dict['x'])
        self.cur += self.as_depth[1]
        outs.append(token_dict)

        x = self.recover.forward(outs)

        for i in range(int((self.total_depth - sum(self.as_depth)) / sum(self.depths))):
            outs = []
            H = self.patch_num
            W = self.patch_num

            for blk in self.stage0:
                x = blk(x, H, W)
            x = self.norm0(x)
            self.cur += self.depths[0]

            B, N, _ = x.shape
            idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
            agg_weight = x.new_ones(B, N, 1)
            token_dict = {'x': x,
                          'token_num': N,
                          'map_size': [H, W],
                          'init_grid_size': [H, W],
                          'idx_token': idx_token,
                          'agg_weight': agg_weight}
            outs.append(token_dict.copy())

            # encoder:stage1
            token_dict = self.ctm1(token_dict, as_out, ctm_stage=i + 2)
            # token_dict = self.stage1(token_dict)
            for j, blk in enumerate(self.stage1):
                token_dict = blk(token_dict)
            token_dict['x'] = self.norm1(token_dict['x'])
            self.cur += self.depths[1]
            outs.append(token_dict)

            x = self.recover.forward(outs)

        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)



        return x, as_out


class TransformerDown_DTMFormerV2(nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=192, total_depth=12,
            patch_size=16, embed_dims=[48, 96, 192], num_heads=[2, 4, 8], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size/patch_size)


        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )


        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.ctm1 = ATM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = ATM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])



    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm1(token_dict, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = self.ctm2(token_dict, attnScore, ctm_stage=2)
        for blk in self.stage2:
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)



        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict = blk(token_dict)
        x = self.norm4(token_dict['x'])


        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)



        return x, attnScore




class TransformerDown_DTMFormerV2_attnloss(nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=192, total_depth=12,
            patch_size=16, embed_dims=[48, 96, 192], num_heads=[2, 4, 8], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size/patch_size)


        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )


        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.ctm1 = ATM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = ATM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])



    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        attns = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm1(token_dict, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = self.ctm2(token_dict, attnScore, ctm_stage=2)
        for blk in self.stage2:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)



        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)

        x = self.norm4(token_dict['x'])


        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)



        return x, attnScore, attns


class TransformerDown_DTMFormerV2_attnlossDS(nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=192, total_depth=12,
            patch_size=16, embed_dims=[48, 96, 192], num_heads=[4, 6, 8], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size / patch_size)

        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.ctm1 = ATM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = ATM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])

        # 深监督解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels // 2, out_channels // 4, kernel_size=4, stride=2, padding=1),
            # 32x32 -> 64x64
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels // 4, out_channels // 8, kernel_size=4, stride=2, padding=1),
            # 64x64 -> 128x128
            nn.BatchNorm2d(out_channels // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels // 8, classes, kernel_size=4, stride=2, padding=1)  # 128x128 -> 256x256
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        attns = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict = self.ctm1(token_dict, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = self.ctm2(token_dict, attnScore, ctm_stage=2)
        for blk in self.stage2:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)

        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)

        x = self.norm4(token_dict['x'])

        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)

        x_ds = self.decoder(x)

        return x, attnScore, attns, x_ds

class TransformerDown_DTMFormerV2_attnlossDSTS(nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=192, total_depth=12,
            patch_size=16, embed_dims=[48, 96, 192], num_heads=[4, 6, 8], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size / patch_size)

        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.ctm1 = ATM_savetokenscore(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = ATM_savetokenscore(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])

        # 深监督解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels // 2, out_channels // 4, kernel_size=4, stride=2, padding=1),
            # 32x32 -> 64x64
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels // 4, out_channels // 8, kernel_size=4, stride=2, padding=1),
            # 64x64 -> 128x128
            nn.BatchNorm2d(out_channels // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels // 8, classes, kernel_size=4, stride=2, padding=1)  # 128x128 -> 256x256
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        attns = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict, token_weight1, idx_cluster1  = self.ctm1(token_dict, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict, token_weight2, idx_cluster2 = self.ctm2(token_dict, attnScore, ctm_stage=2)
        for blk in self.stage2:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)

        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)

        x = self.norm4(token_dict['x'])

        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)

        x_ds = self.decoder(x)

        token_weights = [token_weight1, token_weight2]
        idx_clusters = [idx_cluster1, idx_cluster2]
        return x, attnScore, attns, x_ds, token_weights, idx_clusters


class TransformerDown_DTMFormerV2VarianceStageDict(nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=192, total_depth=12,
            patch_size=16, embed_dims=[48, 96, 192], num_heads=[4, 4, 4], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size / patch_size)

        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.atm1 = ATM_VarianceStageDict(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.atm2 = ATM_VarianceStageDict(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])


    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        attns = []
        attnScores = []
        variances = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)
        attnScores.append(attnScore)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict, token_weight1, idx_cluster1, variance = self.atm1(token_dict, outs, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)
        variances.append(variance)


        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_vals = torch.min(attn_map, dim=1, keepdim=True)[0]
        max_vals = torch.max(attn_map, dim=1, keepdim=True)[0]  # 形状为 [b, 1]
        attnScore1 = (attn_map - min_vals) / (max_vals - min_vals + 1e-8)
        attnScore = var_downup(attnScore1, outs[0], outs[1])
        attnScore = rearrange(attnScore, 'b (h w) 1-> b h w', h=self.patch_num, w=self.patch_num)
        attnScores.append(attnScore)

        token_dict, token_weight2, idx_cluster2, variance = self.atm2(token_dict, outs, attnScore1, ctm_stage=2)
        for blk in self.stage2:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)
        variances.append(variance)

        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)

        x = self.norm4(token_dict['x'])

        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)



        return x, attnScores, attns, outs, variances


class TransformerDown_DTMFormerV2FirstStageChoose(nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=192, total_depth=12,
            patch_size=16, embed_dims=[48, 96, 192], num_heads=[4, 4, 4], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            # sr_ratios=[4, 2, 1], pretrained=None,
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size / patch_size)

        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[0], self.out_channels),
            Rearrange('b s c -> b c s')
        )

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.atm1 = HTM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.atm2 = HTM(self.sample_ratios, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock_attnloss_tokenscore(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ltr3 = FAT_PTRV2(in_channels=[embed_dims[1], embed_dims[2]], out_channels=embed_dims[1])
        self.stage3 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm3 = norm_layer(embed_dims[1])

        self.ltr4 = FAT_PTRV2(in_channels=[embed_dims[0], embed_dims[1]], out_channels=embed_dims[0])
        self.stage4 = nn.ModuleList([UTSBlock_attnloss(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm4 = norm_layer(embed_dims[0])


    def forward(self, img):
        x = self.to_patch_embedding(img)

        outs = []
        attns = []
        attnScores = []
        variances = []
        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
            attns.append(attn)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:,0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_vals = torch.min(attn_map, dim=1, keepdim=True)[0]
        max_vals = torch.max(attn_map, dim=1, keepdim=True)[0]  # 形状为 [b, 1]
        attnScore0 = (attn_map - min_vals) / (max_vals - min_vals + 1e-8)
        attnScore = rearrange(attnScore0, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)
        attnScores.append(attnScore)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        token_dict, token_weight1, idx_cluster1, variance, score1 = self.atm1(token_dict, outs, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)
        variances.append(variance)

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_vals = torch.min(attn_map, dim=1, keepdim=True)[0]
        max_vals = torch.max(attn_map, dim=1, keepdim=True)[0]  # 形状为 [b, 1]
        attnScore1 = (attn_map - min_vals) / (max_vals - min_vals + 1e-8)
        attnScore = var_downup(attnScore1, outs[0], outs[1])
        attnScore = rearrange(attnScore, 'b (h w) 1-> b h w', h=self.patch_num, w=self.patch_num)
        attnScores.append(attnScore)

        token_dict, token_weight2, idx_cluster2, variance, score2 = self.atm2(token_dict, outs, attnScore1, ctm_stage=2)
        for blk in self.stage2:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)
        variances.append(variance)

        token_dict = outs[1]
        token_dict['x'] = self.ltr3([outs[1], outs[2]])
        for blk in self.stage3:
            token_dict, attn = blk(token_dict)
            attns.append(attn)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        token_dict = outs[0]
        token_dict['x'] = self.ltr4([outs[0], outs[3]])
        for blk in self.stage4:
            token_dict, attn = blk(token_dict)
            attns.append(attn)

        x = self.norm4(token_dict['x'])

        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)



        return x, attnScores, attns, outs, variances, [score1,score2]


class TransformerDown_DTMFormerV3(nn.Module):
    def __init__(
            self, in_channels=3, image_size=256, out_channels=192, total_depth=12,
            patch_size=16, embed_dims=[48, 96, 192], num_heads=[2, 4, 8], mlp_ratios=[4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[2, 2, 4],
            sr_ratios=[1, 1, 1], pretrained=None,
            k=5, sample_ratios=0.25, classes=4,
            return_map=False, ):
        super().__init__()

        self.total_depth = total_depth
        self.depths = depths
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map
        self.in_channels = in_channels
        self.k = k
        self.classes = classes
        self.patch_num = int(image_size/patch_size)


        self.out_channels = out_channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dims[0]),
        )
        self.from_patch_embedding = nn.Sequential(
            nn.Linear(embed_dims[2], self.out_channels),
            Rearrange('b s c -> b c s')
        )


        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0

        self.stage0_as = nn.ModuleList([UTSTransformer(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=1)
            for j in range(self.depths[0])])
        self.norm0_as = norm_layer(embed_dims[0])

        self.ctm1 = ATM(self.sample_ratios, self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([UTSBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(self.depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = ATM(self.sample_ratios/4, self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([UTSBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(self.depths[2])])
        self.norm2 = norm_layer(embed_dims[2])




    def forward(self, img):
        x = self.to_patch_embedding(img)

        for blk in self.stage0_as:
            x, attn = blk(x, self.patch_num, self.patch_num)
        x = self.norm0_as(x)
        self.cur += self.depths[0]

        attn_map = attn[:, 0:2].sum(dim=1).sum(dim=1)
        b = attn_map.shape[0]
        n = attn_map.shape[1]
        min_as_out = torch.min(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        max_as_out = torch.max(attn_map, dim=1)[0].repeat(n).reshape(n, b).transpose(0, 1)
        attnScore = (attn_map[:, ] - min_as_out) / (max_as_out - min_as_out)
        attnScore = rearrange(attnScore, 'b (h w)-> b h w', h=self.patch_num, w=self.patch_num)

        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [self.patch_num, self.patch_num],
                      'init_grid_size': [self.patch_num, self.patch_num],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}

        down_dict, token_dict = self.ctm1(token_dict, attnScore, ctm_stage=1)
        for j, blk in enumerate(self.stage1):
            token_dict = blk([token_dict, down_dict])
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]

        down_dict, token_dict = self.ctm2(token_dict, attnScore, ctm_stage=2)
        for blk in self.stage2:
            token_dict = blk([token_dict, down_dict])
        x = self.norm2(token_dict['x'])
        self.cur += self.depths[2]


        x = self.from_patch_embedding(x)  # b c h*w
        b, _, _ = x.size()
        x = x.view(b, self.out_channels, self.patch_num, self.patch_num)



        return x, attnScore



class Transformer_sparse1(nn.Module):
    def __init__(
            self, in_channels, out_channels, image_size, depth=12, dmodel=1024, mlp_dim=2048,
            patch_size=16, heads=12, dim_head=64, emb_dropout=0.1,

            embed_dims=[24, 48, 96, 192],  num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.1, drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, depths=[3, 3, 3, 3], sr_ratios=[8, 4, 2, 1], num_stages=4,
            k=5, sample_ratios=[0.25, 0.25, 0.25],
            return_map=False):
        super().__init__()

        # 第一部分
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width

        self.dmodel = embed_dims[0]

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )
        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
        )


        # 第二部分
        self.depths = depths
        self.embed_dims = embed_dims
        self.sample_ratios = sample_ratios
        self.k = k
        # self.recover = MTA_FAT1()
        self.recover = MTA_FAT1_light()

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0
        self.stage0 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for j in range(depths[0])])
        self.norm0 = norm_layer(embed_dims[0])

        self.ctm1 = CTM(self.sample_ratios[0], self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([TCBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = CTM(self.sample_ratios[1], self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([TCBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ctm3 = CTM(self.sample_ratios[2], self.embed_dims[2], self.embed_dims[3], self.k)
        self.stage3 = nn.ModuleList([TCBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for j in range(depths[3])])
        self.norm3 = norm_layer(embed_dims[3])



    def forward(self, img):
        # torch.Size([4, 3, 256, 256])
        x = self.to_patch_embedding(img)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        # torch.Size([4, 256, 24])
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)


        outs = []
        H = int(math.sqrt(n))
        W = H

        # encoder:stage0
        # x = self.stage1(x)
        for blk in self.stage0:
            x = blk(x, H, W)
        x = self.norm0(x)
        self.cur += self.depths[0]

        # init token dict
        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        # encoder:stage1
        token_dict = self.ctm1(token_dict)
        # token_dict = self.stage1(token_dict)
        for j, blk in enumerate(self.stage1):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        # encoder:stage2
        token_dict = self.ctm2(token_dict)
        # token_dict = self.stage2(token_dict)
        for j, blk in enumerate(self.stage2):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)

        # encoder:stage3
        token_dict = self.ctm3(token_dict)
        # token_dict = self.stage3(token_dict)
        for j, blk in enumerate(self.stage3):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[3]
        outs.append(token_dict)

        # MTA and recover
        x = self.recover.forward(outs)  # torch.Size([4, 1024, 128])


        # torch.Size([4, 256, 192])
        x = self.recover_patch_embedding(x)
        # torch.Size([4, 192, 16, 16])
        return x


class Transformer_sparse2(nn.Module):
    def __init__(
            self, in_channels, out_channels, image_size, depth=12, dmodel=1024, mlp_dim=2048,
            patch_size=16, heads=12, dim_head=64, emb_dropout=0.1,

            embed_dims=[48, 96, 192, 384],  num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.1, drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, depths=[3, 3, 3, 3], sr_ratios=[8, 4, 2, 1], num_stages=4,
            k=5, sample_ratios=[0.25, 0.25, 0.25],
            return_map=False):
        super().__init__()

        # 第一部分
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width

        self.dmodel = embed_dims[0]

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        self.dropout = nn.Dropout(emb_dropout)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(self.patch_dim, self.dmodel),
        )
        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),
        )


        # 第二部分
        self.depths = depths
        self.embed_dims = embed_dims
        self.sample_ratios = sample_ratios
        self.k = k
        self.recover = MTA_FAT2()

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cur = 0
        self.stage0 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for j in range(depths[0])])
        self.norm0 = norm_layer(embed_dims[0])

        self.ctm1 = CTM(self.sample_ratios[0], self.embed_dims[0], self.embed_dims[1], self.k)
        self.stage1 = nn.ModuleList([TCBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for j in range(depths[1])])
        self.norm1 = norm_layer(embed_dims[1])

        self.ctm2 = CTM(self.sample_ratios[1], self.embed_dims[1], self.embed_dims[2], self.k)
        self.stage2 = nn.ModuleList([TCBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for j in range(depths[2])])
        self.norm2 = norm_layer(embed_dims[2])

        self.ctm3 = CTM(self.sample_ratios[2], self.embed_dims[2], self.embed_dims[3], self.k)
        self.stage3 = nn.ModuleList([TCBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[self.cur + j], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for j in range(depths[3])])
        self.norm3 = norm_layer(embed_dims[3])



    def forward(self, img):
        # torch.Size([4, 3, 256, 256])
        x = self.to_patch_embedding(img)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        # torch.Size([4, 256, 24])
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)



        outs = []
        H = int(math.sqrt(n))
        W = H

        a = x

        # encoder:stage0
        # x = self.stage1(x)
        for blk in self.stage0:
            x = blk(x, H, W)
        x = self.norm0(x)
        self.cur += self.depths[0]

        # init token dict
        B, N, _ = x.shape
        idx_token = torch.arange(N)[None, :].repeat(B, 1).cuda()
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        outs.append(token_dict.copy())

        # encoder:stage1
        token_dict = self.ctm1(token_dict)
        # token_dict = self.stage1(token_dict)
        for j, blk in enumerate(self.stage1):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm1(token_dict['x'])
        self.cur += self.depths[1]
        outs.append(token_dict)

        # encoder:stage2
        token_dict = self.ctm2(token_dict)
        # token_dict = self.stage2(token_dict)
        for j, blk in enumerate(self.stage2):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm2(token_dict['x'])
        self.cur += self.depths[2]
        outs.append(token_dict)

        # encoder:stage3
        token_dict = self.ctm3(token_dict)
        # token_dict = self.stage3(token_dict)
        for j, blk in enumerate(self.stage3):
            token_dict = blk(token_dict)
        token_dict['x'] = self.norm3(token_dict['x'])
        self.cur += self.depths[3]
        outs.append(token_dict)

        # MTA and recover
        x = self.recover.forward(outs)  # torch.Size([4, 1024, 128])


        # torch.Size([4, 256, 192])
        x = self.recover_patch_embedding(x)
        # torch.Size([4, 192, 16, 16])
        return x