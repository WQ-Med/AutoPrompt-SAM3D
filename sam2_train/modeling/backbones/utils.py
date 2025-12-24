# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Some utilities for backbones, in particular for windowing"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

#以这个文件为主函数时需要
import sys
sys.path.append('/home/data/Medical-SAM2-main/')
#以这个文件为主函数时需要

from sam2_train.modeling.sam.prompt_encoder import PromptEncoder
from sam2_train.modeling.sam2_utils import LayerNorm2d


from typing import Optional, Tuple, Type

def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.
    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.
    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(
        B, Hp // window_size, Wp // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, ...] = (7, 7),
        stride: Tuple[int, ...] = (4, 4),
        padding: Tuple[int, ...] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) #float32
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x



#自己写的
#各种adapter适应器的代码生成
class DWConv(nn.Module):
    """Depthwise Conv + Conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, act_layer=nn.GELU):
        super().__init__()
        self.dconv = nn.Conv2d( #逐深度卷积
            in_channels, in_channels, kernel_size=kernel_size,padding=kernel_size//2, #保证卷积操作不会改变输入的空间尺寸
            stride=stride, groups=in_channels
        )
        self.pconv = nn.Conv2d( #逐点卷积
            in_channels, out_channels, kernel_size=1,
            stride=1, groups=1
        )
 
    def forward(self, x):
        #  # 改变 x 的维度顺序从 [batch, height, width, channels] -> [batch, channels, height, width]
        x = x.permute(0, 3, 1, 2)  # 转换为 [batch, channels, height, width]
        x = self.dconv(x)
        x = self.pconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        return x


class DWConvAdapter(nn.Module):
    def __init__(self, dim, mlp_ratio=0.25, act_layer=nn.GELU, kernel_size = 3):
        """
        深度可分离卷积适配器，包含卷积、全连接和归一化处理。
        
        :param dim: 输入的特征维度
        :param expansion_factor: 扩展因子，决定中间层的通道数
        :param kernel_size: 深度可分离卷积的卷积核大小
        """
        super(DWConvAdapter, self).__init__()
        D_hidden_features = int(dim * mlp_ratio)
        # 1. FC Down - 降维
        self.fc_down = nn.Linear(dim, D_hidden_features)
        
        # 2. 激活函数
        self.act= act_layer()
        
        # 3. 深度可分离卷积
        self.dwconv = DWConv(in_channels=D_hidden_features,out_channels=D_hidden_features,kernel_size=kernel_size)
        
        # 4. 层归一化
        self.layer_norm = nn.LayerNorm(D_hidden_features)
        
        # 6. FC Up - 恢复维度
        self.fc_up = nn.Linear(D_hidden_features, dim)

    def forward(self, x):
        """
        前向传播过程：
        1. 降维 -> 激活 -> 深度卷积 -> 归一化 -> 激活 -> 升维
        2. 残差连接
        """
        residual = x 
        x = self.fc_down(x)
        x = self.act(x)
        xs = self.act(self.layer_norm(self.dwconv(x)))
        x = self.fc_up(x + xs)
        return x + residual
    

class CNNAdapter(nn.Module):
    def __init__(self, dim, mlp_ratio=0.25, act_layer=nn.GELU, kernel_size = 3):
        super(CNNAdapter, self).__init__()
        D_hidden_features = int(dim * mlp_ratio)
        # 1. FC Down - 降维
        self.fc_down = nn.Linear(dim, D_hidden_features)
        # 2. 层归一化
        self.layer_norm = nn.LayerNorm(D_hidden_features)
        # 3. 深度可分离卷积
        self.dwconv = DWConv(in_channels=D_hidden_features,out_channels=D_hidden_features,kernel_size=kernel_size)
        # 4. 激活函数
        self.act= act_layer()
        # 5. FC UP - 恢复维度
        self.fc_up = nn.Linear(D_hidden_features, dim)

    def forward(self,x):
        x = x.permute(0, 2, 3, 1)
        residual = x
        x = self.fc_down(x)
        x = self.layer_norm(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc_up(x)
        return (residual + x).permute(0, 3, 1, 2)


class Adapter(nn.Module):
    def __init__(self, dim, mlp_ratio=0.25, act_layer=nn.GELU) -> None:
        super(Adapter,self).__init__()
        D_hidden_features = int(dim * mlp_ratio)
        # 1. FC Dowm - 降维
        self.fc_down = nn.Linear(dim,D_hidden_features)
        # 2.激活函数
        self.act = act_layer()
        # 3. FC Up -升维
        self.fc_up = nn.Linear(D_hidden_features,dim)
    def forward(self,x):
        residual = x
        x = self.fc_down(x)
        x = self.act(x)
        x = self.fc_up(x)
        return residual + x

class ParallelAdapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

class Confidence_MLP(nn.Module):
    def __init__(self,in_channels,hidden_dim) -> None:
        super(Confidence_MLP,self). __init__()
        self.mlp = nn.Sequential(
            # nn.Linear(in_channels,1),
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出一个标量
        )
    def forward(self, F):
        B, C, H, W = F.shape
        F = F.permute(0, 2, 3, 1).contiguous()  # 保证内存连续性
        F_flat = F.view(-1, C)                 # 等价 reshape，但保持可导
        S_flat = self.mlp(F_flat)
        S = S_flat.view(B, H, W).unsqueeze(1)
        return S

# class Confidence_MLP(nn.Module):
#     def __init__(self, in_channels, hidden_dim=64):
#         super(Confidence_MLP, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(hidden_dim, 1, kernel_size=1)  # 输出单通道 sal_map
#         )

#     def forward(self, x):
#         return self.net(x)  # 输出 shape: [B, 1, H, W]
    
class DualAttentionPrompter(nn.Module):
    def __init__(self, in_channels, inter_channels=None,target_size=(512,512)):
        super(DualAttentionPrompter, self).__init__()
        if inter_channels is None:
            inter_channels = in_channels // 2

        #可学习位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, in_channels, *target_size))

        # 空间注意力模块
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 通道注意力模块（简化 SE block）
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 输出显著图
        self.out_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # print("F stats:", x.min().item(), x.max().item(), x.mean().item())
        # print("F contains nan:", torch.isnan(x).any())

        x = F.interpolate(x, size=(512,512), mode='bilinear', align_corners=False)
        x = x + self.pos_embed
        sa = self.spatial_attn(x) * x
        ca = self.channel_attn(x) * x
        fused = sa + ca
        sal_map = self.out_conv(fused)
        return sal_map  # shape: [B, 1, H, W]

def double_conv(in_channels, out_channels):  # 双层卷积模型，神经网络最基本的框架
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),  # 加入Bn层提高网络泛化能力（防止过拟合），加收敛速度
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),  # 3指kernel_size，即卷积核3*3
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

#新Unet解码器尝试
'''
class MultiScaleSaliencyDecoder(nn.Module):
    # feat32 #(B,256,32,32)
    # feat63 #(B,64,64,64)
    # feat128 #(B,32,128,128)
    def __init__(self,outsize=(512,512)) -> None:
        super().__init__()
        self.dconv_up32 = double_conv(256,128)
        self.dconv_up64 = double_conv(64+64,64)
        self.dconv_up128 = double_conv(64,32)
        self.upsample2 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.upsample1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.upsample0 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)
        self.out = nn.Conv2d(16,1,1)
        self.final_upsample = nn.Upsample(size=outsize, mode='bilinear', align_corners=False)
    
    def forward(self,feat128,feat64,feat32):
        x = self.dconv_up32(feat32) # 128x32x32
        x= self.upsample2(x) # 64x64x64
        x =torch.cat([x,feat64],dim=1) # 128x64x64

        x = self.dconv_up64(x) # 64x64x64
        x = self.upsample1(x)  # 32x128x128
        x = torch.cat([x, feat128], dim=1)  # 64x128x128

        x = self.dconv_up128(x)  # 32x128x128
        x = self.upsample0(x)   # 16x256x256
        out = self.out(x)  # 1x256x256

        return self.final_upsample(out)
'''

#原来的Unet结构
class MultiScaleSaliencyDecoder(nn.Module):
    def __init__(self, ch_128=32, ch_64=64, ch_32=256, mid_channels=128, out_size=(512, 512)):
        super().__init__()

        self.up_conv32 = nn.Sequential(
            nn.Conv2d(ch_32, mid_channels, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)  # 32→128
        )

        self.up_conv64 = nn.Sequential(
            nn.Conv2d(ch_64, mid_channels, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 64→128
        )

        self.up_conv128 = nn.Sequential(
            nn.Conv2d(ch_128, mid_channels, 3, padding=1),
            nn.ReLU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 3, mid_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, 1, 1)
        )

        self.final_upsample = nn.Upsample(size=out_size, mode='bilinear', align_corners=False)


    def forward(self, feat_128, feat_64, feat_32):
        dfeat_128,dfeat_64,dfeat_32 = feat_128.detach(),feat_64.detach(),feat_32.detach() #梯度阶段，这部分的梯度不会影响image_encoder
        f1 = self.up_conv128(dfeat_128)  # 128x128
        f2 = self.up_conv64(dfeat_64)    # 64→128
        f3 = self.up_conv32(dfeat_32)    # 32→128

        fused = torch.cat([f1, f2, f3], dim=1)  # (B, mid_channels*3, 128, 128)
        out = self.fuse(fused)                 # (B, 1, 128, 128)
        out = self.final_upsample(out)         # (B, 1, 512, 512)
        return out

class ImagePromptEncoder(PromptEncoder):

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        ) -> None:
        super().__init__(embed_dim,image_embedding_size,input_image_size,mask_in_chans,activation = nn.GELU,)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(3, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        batch_size = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        if batch_size == -1:
            bs = self._get_batch_size(points, boxes, masks)
        else:
            bs = batch_size
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class ClSHead(nn.Module):
    def __init__(self,hidden_dim) -> None:
        super().__init__()
        # self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1,1))
        self.output = nn.Sequential(
            nn.Linear(hidden_dim,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )
        # self.output = nn.Sequential(
        #     nn.Linear(hidden_dim, 256),  # 输入维度是展平后的特征维度
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)  # 输出一个标量，用于二分类（是否有前景）
        # )
    def forward(self,hig_res_feature):
        hig_res_feature = self.global_max_pool(hig_res_feature) #结果形状为(B,C,1,1)
        flattened_features = hig_res_feature.view(hig_res_feature.size(0), -1)  
        cls_logits = self.output(flattened_features)
        return cls_logits

class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPHead, self).__init__()
        
        # 定义 MLP 中的全连接层
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一个全连接层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 第二个全连接层
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 输出层
        
        # ReLU 激活函数
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.reshape(x.size(0), -1)  # Flatten 特征图 (batch_size, 128, 32, 32) -> (batch_size, 128*32*32)
        x = self.relu(self.fc1(x))  # 第一层全连接，ReLU 激活
        x = self.relu(self.fc2(x))  # 第二层全连接，ReLU 激活
        x = self.fc3(x)  # 输出层（没有激活函数，输出直接用于二分类）
        return x

class MaskClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 输入 [B, 1, 512, 512]
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 256, 256]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 128, 128]
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [B, 64, 64, 64]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 64, 32, 32]
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [B, 128, 16, 16]
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 128, 8, 8]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x #返回的是未经过sigmoid的logits

if __name__ == '__main__':
    sam_prompt_embed_dim = 256
    sam_image_embedding_size = 32
    image_size = 512
    Imageprompt_encoder = ImagePromptEncoder(
        embed_dim=sam_prompt_embed_dim,
            image_embedding_size=(
                sam_image_embedding_size,
                sam_image_embedding_size,
            ),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
    Imageprompt_encoder(points=None,boxes=None,masks=torch.rand(1,3,128,128))
    #sparse_embeddings [1,0,256]
    #dense_embeddings   [1,256,128,128]
