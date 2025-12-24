import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class TripleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.triple_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.triple_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is NCHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class IoUHead(nn.Module):
    def __init__(
        self,
        feature_dim=64,
        hidden_dim=128,
        num_layers=2,              # 支持任意层数
        activation=nn.ReLU,
    ):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 构建 MLP
        h = [hidden_dim] * (num_layers - 1)
        layers = []
        for in_dim, out_dim in zip([feature_dim] + h, h + [1]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != 1:  # 最后一层不加激活
                layers.append(activation())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.global_pool(x)  # [N, 64, 1, 1]
        x = x.flatten(1)         # [N, 64]
        x = self.mlp(x)          # [N, 1]
        return torch.sigmoid(x)  # [N, 1]，Sigmoid放外面更灵活

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        n_filters = [32, 64, 128, 256, 512]

        self.inc = DoubleConv(n_channels, n_filters[0])
        self.down1 = Down(n_filters[0], n_filters[1])
        self.down2 = Down(n_filters[1], n_filters[2])
        self.down3 = Down(n_filters[2], n_filters[3])
        self.down4 = Down(n_filters[3], n_filters[4])
        self.up1 = Up(n_filters[4], n_filters[3])
        self.up2 = Up(n_filters[3], n_filters[2])
        self.up3 = Up(n_filters[2], n_filters[1])
        self.up4 = Up(n_filters[1], n_filters[0])
        self.outc = OutConv(n_filters[0], n_classes)

        # #辅助分类器
        # self.aux_classifier1 = nn.Conv2d(512, n_classes, kernel_size=1)  # P5层
        self.aux_classifier2 = nn.Conv2d(128, n_classes, kernel_size=1)  # P4层
        self.aux_classifier3 = nn.Conv2d(64, n_classes, kernel_size=1)  # P3层

        #新增输出头
        self.iou_head = IoUHead(feature_dim=32)

    def forward(self, x):
        x1 = self.inc(x)    #N,32,512,512
        x2 = self.down1(x1) #N,64,256,256
        x3 = self.down2(x2) #N,128,128,128
        x4 = self.down3(x3) #N,256,64,64
        x5 = self.down4(x4) #N,512,32,32
        x = self.up1(x5, x4)    #N,256,64,64
        x = self.up2(x, x3)     #N,128,128,128
        aux_out2 = self.aux_classifier2(x) #N,1,128,128
        x = self.up3(x, x2)     #N,64,256,256
        aux_out3 = self.aux_classifier3(x) #N,1,256,256
        x = self.up4(x, x1)     #N,32,512,512
        logits = self.outc(x)   #N,1,512,512
        iou_out = self.iou_head(x) #N,1
        print("logits stats:", logits.min().item(), logits.max().item(), logits.mean().item())
        # prob_map = torch.sigmoid(logits)
        # print("prob_map mean:", prob_map.mean().item())

        return logits, aux_out2, aux_out3 , iou_out
    
