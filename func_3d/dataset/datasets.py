from torch.utils.data import DataLoader, Dataset
from .base_dataset import BaseVolumeDataset,flatten_collate_fn #使用train.py为main函数时使用
# from base_dataset import BaseVolumeDataset,flatten_collate_fn #当dasets.py为main函数时使用
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

class ColonVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-57, 175)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 65.175035
        self.global_std = 32.651197
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 1
        self.img_size = (512,512)

class LiTSVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-48, 163)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 60.057533
        self.global_std = 40.198017
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2
        self.img_size = (512,512)

class KiTSVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-54, 247)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 59.53867
        self.global_std = 55.457336
        self.spatial_index = [0, 1, 2]  # index used to convert to DHW
        self.do_dummy_2D = False
        self.target_class = 2

class PancreasVolumeDataset(BaseVolumeDataset):
    def _set_dataset_stat(self):
        # org load shape: d, h, w
        self.intensity_range = (-39, 204)
        self.target_spacing = (1, 1, 1)
        self.global_mean = 68.45214
        self.global_std = 63.422806
        self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        self.do_dummy_2D = True
        self.target_class = 2

def save_t_slices_with_plt(images, label_tensor, save_dir, prefix="slice", cmap="gray"):
    """
    保存 tensor 中的 T 个时间帧为 2D 图像（使用 matplotlib），并且生成两个子图
    - 第一个子图是原始图像。
    - 第二个子图是带标签的图像。
    
    参数:
        images: torch.Tensor [B, C, T, H, W]  (批量大小，通道数，时间步数，图像高，图像宽)
        label_tensor: torch.Tensor [B, T, H, W]  (标签张量，表示每个帧的标签)
        save_dir: 图片保存文件夹
        prefix: 文件名前缀
        cmap: 显示用的颜色映射，比如 'gray'、'hot'、'viridis'
    """
    os.makedirs(save_dir, exist_ok=True)

    # 取第一个 batch、第一通道 → [C,T, H, W]
    slices = images[0].detach().cpu().float()  # 取第一个批次，第一通道（C=0），形状为 [T, H, W]
    label_slices = label_tensor[0].detach().cpu()  # 取第一个批次的标签，形状为 [T, H, W]
    print(torch.max(slices),torch.min(slices))
    print(torch.max(label_slices),torch.min(label_slices))

    for t in range(slices.shape[1]):
        img = slices[:, t]  # 当前时间步的图像，形状为 [C,H, W]
        label_mask = label_slices[t]  # 当前时间步的标签，形状为 [C,H, W]

        # 将多个通道合并成 RGB 图像
        img_rgb = np.moveaxis(img.numpy(), 0, -1)  # 将形状从 [C, H, W] 转换为 [H, W, C]

        # # 可选：归一化（可视化更清晰）
        # img_min, img_max = img_rgb.min(), img_rgb.max()
        # img_rgb_norm = (img_rgb - img_min) / (img_max - img_min + 1e-5)

        # 创建带标签的图像（将标签区域标记为红色）
        label_overlay = np.copy(img_rgb)  #label_overlay = np.copy(img_rgb_norm)
        label_overlay[label_mask == 1] = [255, 0, 0]  # 将目标区域标记为红色（RGB: [1, 0, 0]）

        # 创建两个子图：原图和带标签的图像
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # 两个子图（1行2列）
        
        # 显示原图
        axes[0].imshow(img_rgb)  #axes[0].imshow(img_rgb_norm)
        axes[0].set_title(f"Original Image (Timestep {t})")
        axes[0].axis('off')

        # 显示带标签的图像
        axes[1].imshow(label_overlay)
        axes[1].set_title(f"With Label Mask (Timestep {t})")
        axes[1].axis('off')

        # 保存图像
        filename = os.path.join(save_dir, f"{prefix}_t{t:03d}.png")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    print(f"✅ Saved {slices.shape[0]} 2D slices to: {save_dir}")


if __name__ == '__main__':

    
    # #有目标和无目标都有的dataset处理方式
    
    # colon_dataset = ColonVolumeDataset(path_prefix='data/Task10_Colon',augmentation=True,mode='test')

    # dataloader = DataLoader(dataset=colon_dataset,collate_fn=flatten_collate_fn)

    # for sample in dataloader:
    #     clips = torch.stack([s['clip'] for s in sample])
    #     labels = torch.stack([s["label"] for s in sample])
    #     frame_has_fg = torch.stack([s["frame_has_fg"] for s in sample])
    #     start_idxs = [s["start_idx"] for s in sample]
    #     name = [s["name"] for s in sample]
    #     valid_length = [s['valid_length'] for s in sample]

    #     # save_t_slices_with_plt(clips,save_dir='2D_test')

    #     print(name)
    #     print(clips.shape)
    #     print(labels.shape)
    #     print(frame_has_fg.shape)
    #     print(start_idxs)
    #     print('下一个')
    

    # 只有目标区域的dataset处理方式
    
    colon_dataset = ColonVolumeDataset(path_prefix='data/Task10_Colon',augmentation=True,mode='train',choose='mask')
    dataloader = DataLoader(dataset=colon_dataset)
    for sample in dataloader:
        clips = sample['clip'] #(1,3,23,512,512)
        labels = sample['label']
        start_idxs = sample['start_idx']
        name = sample['name']

        save_t_slices_with_plt(clips,label_tensor=labels,save_dir='2D_test')

        print(name)
        # print(clips.shape)
        # print(labels.shape)
        # print(start_idxs)
        # print('下一个')


    # # 二分类的dataset处理方式
    # def test(mask,label):
    # # 验证 mask 和 label 是否正确对应
    #     for i in range(label.shape[1]):  # 对于每一帧（共有9帧）
    #         if label[0, i] == 1:
    #             # 如果 label 中的第 i 帧有目标（1），则 mask[i] 不应该是全零
    #             if torch.sum(mask[0, i]) == 0:
    #                 print(f"Error: Frame {i} in mask should have a target, but it is all zeros.")
    #                 return False
    #         else:
    #             # 如果 label 中的第 i 帧没有目标（0），则 mask[i] 应该是全零
    #             if torch.sum(mask[0, i]) > 0:
    #                 print(f"Error: Frame {i} in mask should not have a target, but it contains non-zero values.")
    #                 return False
    #     return True
    # colon_dataset = ColonVolumeDataset(path_prefix='data/Task10_Colon',augmentation=True,mode='test',choose='classification')
    # dataloader = DataLoader(dataset=colon_dataset)
    # for sample in dataloader:
    #     clips = sample['clip']
    #     ture_label = sample['true_label']
    #     labels = sample['label']
    #     if test(mask=labels,label=ture_label):
    #         pass
    #     else:
    #         raise ValueError('有问题')
    #     start_idxs = sample['start_idx']
    #     name = sample['name']

    #     # save_t_slices_with_plt(clips,label_tensor=labels,save_dir='2D_test')

    #     print(name)
    #     print(clips.shape)
    #     print(labels.shape)
    #     print(start_idxs)
    #     print('下一个')
    

        