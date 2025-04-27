from ..components.unets_parts import *

class CNNs(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=False):
        super(CNNs, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.scale = 1  # 1 2 4

        self.inc = DoubleConv(n_channels, 64 // self.scale)
        self.down1 = Down(64 // self.scale, 128 // self.scale)
        self.down2 = Down(128 // self.scale, 256 // self.scale)
        self.down3 = Down(256 // self.scale, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return x4


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.scale = 1  # 1 2 4

        self.inc = DoubleConv(n_channels, 64 // self.scale)
        self.down1 = Down(64 // self.scale, 128 // self.scale)
        self.down2 = Down(128 // self.scale, 256 // self.scale)
        self.down3 = Down(256 // self.scale, 512 // self.scale)
        factor = 2 if bilinear else 1
        self.down4 = Down(512 // self.scale, 1024 // factor // self.scale)
        self.up1 = Up(1024 // self.scale, 512 // factor // self.scale, bilinear)
        self.up2 = Up(512 // self.scale, 256 // factor // self.scale, bilinear)
        self.up3 = Up(256 // self.scale, 128 // factor // self.scale, bilinear)
        self.up4 = Up(128 // self.scale, 64 // self.scale, bilinear)
        self.outc = OutConv(64 // self.scale, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ResUnet(nn.Module):
    def __init__(self, n_channels, n_classes, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()
        self.scale = 1  # 1 2 4
        self.filters = [x//self.scale for x in filters]

        self.input_layer = nn.Sequential(
            nn.Conv2d(n_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(nn.Conv2d(n_channels, filters[0], kernel_size=3, padding=1))
        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)
        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)
        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)
        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)
        self.output_layer = nn.Conv2d(filters[0], n_classes, 1, 1)

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up_residual_conv1(x5)
        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up_residual_conv2(x7)
        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        x10 = self.up_residual_conv3(x9)
        output = self.output_layer(x10)
        return output


class AttUnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(AttUnet, self).__init__()

        self.scales = 1
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = DoubleConv(n_channels, 64//self.scales)
        self.Conv2 = DoubleConv(64//self.scales, 128//self.scales)
        self.Conv3 = DoubleConv(128//self.scales, 256//self.scales)
        self.Conv4 = DoubleConv(256//self.scales, 512//self.scales)
        self.Conv5 = DoubleConv(512//self.scales, 1024//self.scales)

        self.Up5 = up_conv(1024//self.scales, 512//self.scales)
        self.Att5 = Attention_block(F_g=512//self.scales, F_l=512//self.scales, F_int=256//self.scales)
        self.Up_conv5 = DoubleConv(1024//self.scales, 512//self.scales)
        self.Up4 = up_conv(ch_in=512//self.scales, ch_out=256//self.scales)
        self.Att4 = Attention_block(F_g=256//self.scales, F_l=256//self.scales, F_int=128//self.scales)
        self.Up_conv4 = DoubleConv(512//self.scales, 256//self.scales)
        self.Up3 = up_conv(256//self.scales, 128//self.scales)
        self.Att3 = Attention_block(F_g=128//self.scales, F_l=128//self.scales, F_int=64//self.scales)
        self.Up_conv3 = DoubleConv(256//self.scales, 128//self.scales)
        self.Up2 = up_conv(ch_in=128//self.scales, ch_out=64//self.scales)
        self.Att2 = Attention_block(F_g=64//self.scales, F_l=64//self.scales, F_int=32//self.scales)
        self.Up_conv2 = DoubleConv(128//self.scales, 64//self.scales)

        self.Conv_1x1 = nn.Conv2d(64//self.scales, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        return d1