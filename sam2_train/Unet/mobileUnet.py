from  sam2_train.Unet.mobilenet import MobileNet
import torch.nn as nn
from collections import OrderedDict
import torch
import torchsummary as summary

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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.residual = (in_channels == out_channels)

    def forward(self, x):
        if self.residual:
            return x + self.double_conv(x)
        return self.double_conv(x)

def conv2d(filter_in, filter_out, kernel_size, groups=1, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, groups=groups, bias=False)),
        ("bn", nn.InstanceNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(inplace=True)),
    ]))


class mobilenet(nn.Module):
    def __init__(self, n_channels):
        super(mobilenet, self).__init__()
        self.model = MobileNet(n_channels)

    def forward(self, x):
        out3 = self.model.layer1(x)
        out4 = self.model.layer2(out3)
        out5 = self.model.layer3(out4)

        return out3, out4, out5


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


class MobileUNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MobileUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes= n_classes

        '''新增输出头'''
        #对象置信度头(预测是否有目标)
        self.obj_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  #全局平均池化 [N,C,1,1]
            nn.Conv2d(64,1,kernel_size=1),
            nn.ReLU()
        )
        
        #  IOU分数头(预测分割质量)
        '''
        self.iou_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  #全局平均池化
            nn.Conv2d(64,1,kernel_size=1),
            nn.ReLU()
        )
        '''

        # IOU分数头2
        self.iou_head = IoUHead(feature_dim=64)

        # ---------------------------------------------------#
        #   64,64,256；32,32,512；16,16,1024
        # ---------------------------------------------------#
        self.backbone = mobilenet(n_channels)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = DoubleConv(1024, 512)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = DoubleConv(1024, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = DoubleConv(512, 128)

        self.up4 = nn.Upsample(scale_factor=2, mode='nearest')
        #nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv4 = DoubleConv(128, 64)

        self.oup = nn.Conv2d(64, n_classes, kernel_size=1)

        # #辅助分类器
        self.aux_classifier1 = nn.Conv2d(512, n_classes, kernel_size=1)  # P5层
        self.aux_classifier2 = nn.Conv2d(256, n_classes, kernel_size=1)  # P4层
        self.aux_classifier3 = nn.Conv2d(128, n_classes, kernel_size=1)  # P3层

    def forward(self, x):
        #  backbone
        x2, x1, x0 = self.backbone(x) #x0.dtype就变成了float16  #x2: N*256*128*128  x1: N*512*64*64 x0: N*1024*32*32
        # print(f"x2.shape: {x2.shape}, x1: {x1.shape}, x0: {x0.shape} ")

        P5 = self.up1(x0)
        P5 = self.conv1(P5)           # P5: N*512*64*64

        aux1_out = self.aux_classifier1(P5) #N*1*64*64

        # print(P5.shape)l
        P4 = x1                       # P4: 26x26x512
        P4 = torch.cat([P4, P5], axis=1)   # P4(堆叠后): 26x26x1024
        # print(f"cat 后是： {P4.shape}")
        P4 = self.up2(P4)             # 52x52x1024
        P4 = self.conv2(P4)           # N*256*128*128 #验证期间，P4的dtype为float16

        aux2_out = self.aux_classifier2(P4) ##N*1*128*128

        P3 = x2                       # x2 = 52x52x256
        P3 = torch.cat([P4, P3], axis=1)  # 52x52x512
        P3 = self.up3(P3)
        P3 = self.conv3(P3)  #N*128*256*256

        aux3_out = self.aux_classifier3(P3)  ##N*1*256*256

        P3 = self.up4(P3)
        P3 = self.conv4(P3)   # N*64*512*512
        out = self.oup(P3)
        # print(f"out.shape is {out.shape}")

        #新增输出
        # obj_score = self.obj_head(P3)
        iou_feat = P3.detach()
        iou_score = self.iou_head(iou_feat)

        return out , aux1_out , aux2_out , aux3_out, iou_score