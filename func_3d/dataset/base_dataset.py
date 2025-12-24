#这个文件是6.16开启，主要想做的是将一个完整的3D图像切割成2D切片，直接用于模型训练，而不是又创建一个文件夹来做，这样就没什么意义了
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import nibabel as nib
import numpy as np
import torch.nn.functional as F  
import torch
from typing import Dict, Hashable, Mapping
from monai.transforms import (
    Compose,
    RandCropByPosNegLabeld,
    CropForegroundd,
    SpatialPadd,
    ScaleIntensityRanged,
    RandShiftIntensityd,
    RandFlipd,
    RandAffined,
    RandZoomd,
    RandRotated,
    RandRotate90d,
    RandGaussianNoised,
    RandGaussianSmoothd,
    NormalizeIntensityd,
    MapTransform,
    RandScaleIntensityd,
    RandSpatialCropd,
    Resize,
    EnsureType,
)
from monai.config import KeysCollection
import random

class BinarizeLabeld(MapTransform):
    def __init__(
            self,
            keys: KeysCollection,
            threshold: float = 0.5,
            allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.threshold = threshold

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            if not isinstance(d[key], torch.Tensor):
                d[key] = torch.as_tensor(d[key])

            dtype = d[key].dtype
            d[key] = (d[key] > self.threshold).to(dtype)
        return d

def read_pkl(pkl_path,k,mode):
    file = open(pkl_path,'rb')
    data = pickle.load(file)[k][mode]
    return data


class BaseVolumeDataset(Dataset):
    def __init__(
            self,
            path_prefix,
            augmentation,
            mode='train',
            rand_crop_spatial_size=(100, 512, 512),
            img_size=512,
            k=0,
            do_val_crop=True,
            choose = 'all'):
        super().__init__()
        
        d= read_pkl(pkl_path=os.path.join(path_prefix,'split.pkl'),k=0,mode=mode)
        self.image_files = [os.path.join(path_prefix, d[i][0].strip("/")) for i in list(d.keys())][17:]
        self.label_files= [os.path.join(path_prefix, d[i][1].strip("/")) for i in list(d.keys())][17:]

        self.mode = mode
        self.rand_crop_spatial_size=rand_crop_spatial_size
        self.img_size = img_size
        self.do_val_crop = do_val_crop
        self.choose = choose #选择提取目标数据集还是全部(有无目标都包含)数据集

        self.augmentation = augmentation

        self.intensity_range = (
            self.target_spacing
        ) = (
            self.global_mean
        ) = self.global_std = self.spatial_index = self.do_dummy_2D = self.target_class = None
        
        self._set_dataset_stat()
        self.transforms = self.get_transforms()

    def _set_dataset_stat(self):
        pass
        #----------------------------------只是以colon数据集来测试，所以真正使用的时候一定记得删除---------------------------
        # self.intensity_range = (-57, 175)
        # self.target_spacing = (1, 1, 1)
        # self.global_mean = 65.175035
        # self.global_std = 32.651197
        # self.spatial_index = [2, 1, 0]  # index used to convert to DHW
        # self.do_dummy_2D = True
        # self.target_class = 1

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        print('这是第{}个'.format(index))
        img_path = self.image_files[index]
        print(img_path)
        # img_path = 'data/Task10_Colon/imagesTr/colon_136.nii.gz'
        # img_path = 'data/Task03_LITS17/Training/volume-88/image.nii.gz'
        # print(img_path)
        label_path = self.label_files[index]
        # label_path = 'data/Task10_Colon/labelsTr/colon_136.nii.gz'
        # label_path = 'data/Task03_LITS17/Training/volume-88/segmentation.nii.gz'

        img_vol = nib.load(img_path)
        img = img_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)
        img_spacing = tuple(np.array(img_vol.header.get_zooms())[self.spatial_index])
    
        seg_vol = nib.load(label_path)
        seg = seg_vol.get_fdata().astype(np.float32).transpose(self.spatial_index)

        img[np.isnan(img)] = 0 #使原数据元素为nan值的都变为0
        seg[np.isnan(seg)] = 0
        
        seg = (seg == self.target_class).astype(np.float32) #将seg中所有等于 self.target_class 的像素值转换为浮点数类型的 1，而其他像素值转换为浮点数类型的 0
        if (np.max(img_spacing) / np.min(img_spacing) > 8) or ( #用于医学图像CT/MRI的预处理，处理各向异性的体素间距(是否某个方向的spacing远大于其他方向)
                np.max(self.target_spacing / np.min(self.target_spacing) > 8) #如果是，则使用2D插值，逐切片处理进行重采样，以避免3D插值带来的失真
        ):
            print('对图像进行体素间距处理')
            # resize 2D
            if self.mode != 'nono':
                img_tensor = F.interpolate(
                    input=torch.tensor(img[:, None, :, :]),
                    scale_factor=tuple([img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]),
                    mode="bilinear",
                )
            
                seg_tensor = F.interpolate(
                    input=torch.tensor(seg[:, None, :, :]),
                    scale_factor=tuple(
                        [img_spacing[i] / self.target_spacing[i] for i in range(1, 3)]
                    ),
                    mode="bilinear",
                )
            if self.mode != "nono":
                img = (
                    F.interpolate(
                        input=img_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                        scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )
            
                seg = (
                    F.interpolate(
                        input=seg_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous(),
                        scale_factor=(img_spacing[0] / self.target_spacing[0], 1, 1),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )
        else:
            if self.mode != "nono":
                img = (  #如果没有特别异常的像素间距，那么直接进行整体重采样，目的都是使得图像像素变成target_spacing(1,1,1)
                    F.interpolate( 
                        input=torch.tensor(img[None, None, :, :, :]),
                        scale_factor=tuple(
                            [img_spacing[i] / self.target_spacing[i] for i in range(3)]
                        ),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )
            
                seg = (
                    F.interpolate(
                        input=torch.tensor(seg[None, None, :, :, :]),
                        scale_factor=tuple(
                            [img_spacing[i] / self.target_spacing[i] for i in range(3)]
                        ),
                        mode="trilinear",
                    )
                    .squeeze(0)
                    .numpy()
                )
        # return {'img':img,'label':seg}
        print('经过像素间隔重采样后的图像形状',img.shape) #(53,447,447)
        if (self.augmentation and self.mode == "train") or ((self.do_val_crop and self.mode=='val')):
            trans_dict = self.transforms({"image": img, "label": seg})
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]
        else:
            trans_dict = self.transforms({"image": img, "label": seg})
            img_aug, seg_aug = trans_dict["image"], trans_dict["label"]

        seg_aug = seg_aug.squeeze() #(D,H,W)
        img_aug = img_aug.repeat(3, 1, 1, 1)  #(3,D,H,W)
        
        # return {'img':img_aug,'label':seg_aug}
        return self.extract_clip_slices(image=img_aug,label=seg_aug,name=img_path.split('/')[-1])
    
    def extract_clip_slices(
            self,image, label, name, video_length=12,
    ):
        if self.choose == 'all':
            return self.all_extract_clip_slices(image,label,name,window_size=video_length)
        if self.choose == 'mask':
            return self.mask_extract_clip_slices(image,label,name,video_length)
        if self.choose == 'classification':
            return self.classification_extract_clip_slices(image,label,name,video_length)
        else:
            raise ValueError('没有这个数据处理方式，你的choose:{}是有问题的'.format(self.choose)) 
    
    #打乱有无目标的数据，引发二分类任务
    def classification_extract_clip_slices(
        self, image, label, name, video_length=12
    ):
        '''
        根据所有区域做二分类图片
        '''
        assert video_length % 2 == 0, f"videolegth 必须是偶数，但传入的 video_length={video_length} 是奇数"
        target_indices = torch.where(torch.count_nonzero(label, dim=(1, 2)) > 50)[0]  # 获取包含大于 10 个目标像素的切片索引
                # Step 3: 对每个连续的目标段进行处理
        empty_indices = torch.where(torch.count_nonzero(label, dim=(1, 2)) == 0)[0]   # 无目标的索引
        K = video_length // 2  # 每类抽取 K 个
        # 从有目标的帧中随机选 K 个（如果数量足够）
        if len(target_indices) > K:
            selected_target = target_indices[torch.randperm(len(target_indices))[:K]]
        else:
            selected_target = target_indices  # 如果不足 K 个，则全取

        # 从无目标的帧中随机选 K 个（如果数量足够）
        if len(empty_indices) > K:
            selected_empty = empty_indices[torch.randperm(len(empty_indices))[:K]]
        else:
            selected_empty = empty_indices  # 如果不足 K 个，则全取

        # 合并结果
        segment = torch.cat([selected_target, selected_empty])
        true_label = torch.zeros(len(segment), dtype=torch.int)
        true_label[:len(selected_target)] = 1
        print("最终抽取的索引列表:", segment)
        indices = torch.randperm(len(segment))

        # 使用这个随机索引重新排列 segment 和 label
        shuffled_segment = segment[indices]
        shuffled_label = true_label[indices]

        segment = 0
        clip_frames = []
        label_frames = []
        
        segment_length = len(shuffled_segment)
        if segment_length > video_length:
            # 如果该段长度大于等于 video_length，则随机抽取连续的视频长度的切片
            random_start = random.randint(0, segment_length - video_length)
            random_start = 0
            selected_indices = shuffled_segment[random_start:random_start + video_length]
        else:
            # 如果该段长度小于 video_length，则返回该段所有目标切片
            selected_indices = shuffled_segment
        
        # 提取目标区域的切片和标签
        selected_slices = image[:, selected_indices, :, :]
        selected_labels = label[selected_indices, :, :]
        
        # 将选中的切片和标签转换为2D
        img_2D_transform = Compose([Resize(spatial_size=self.img_size, mode='bilinear')])
        label_2D_transform = Compose([Resize(spatial_size=self.img_size, mode='bilinear')])

        for i in range(selected_slices.shape[1]):
            img_i = selected_slices[:, i]
            label_i = selected_labels[i].unsqueeze(0)
            img_i = img_2D_transform(img_i)
            label_i = label_2D_transform(label_i)
            # 进行阈值化处理，确保标签是二值的 (0 或 1)
            label_i = (label_i > 0.5).float()  # 使用阈值化，值大于0.5的设为1，其余设为0
            # print(torch.nonzero(label_i== 1))
            clip_frames.append(img_i)
            label_frames.append(label_i.squeeze())

        # 堆叠回来
        clip_resized = torch.stack(clip_frames, dim=1)
        label_resized = torch.stack(label_frames, dim=0)

        return {
            "clip": clip_resized,
            'true_label': shuffled_label,
            "label": label_resized,
            "start_idx": selected_indices[0],
            "name": name
        }
   #包含有无目标两个概率 #它的mask需要只为0和1的这个问题并没有得到解决
    def all_extract_clip_slices(
                        self,
                        image,                 # Tensor[C, Z, H, W]
                        label,                 # Tensor[Z, H, W]
                        name,
                        window_size=12,
                        stride=12,
                        min_fg_frames=0,
                        include_background=False
        ):
            """
            从单个 volume 中提取一组滑动窗口 clip(连续帧段)，可用在 Dataset 内部

            返回: List[Dict{
                "clip": Tensor[C, window, H, W],
                "label": Tensor[window, H, W],
                "frame_has_fg": Tensor[window],
                "start_idx": int
            }]
            """
            img_2D_transform = Compose([
                Resize(spatial_size=self.img_size,mode='bilinear'),
                # EnsureType
            ])
            label_2D_transform = Compose([
                Resize(spatial_size=self.img_size,mode='bilinear'),
                # EnsureType
            ])
            # assert image.shape[1:] == label.shape, "Image and label shape mismatch"
            C, Z, H, W = image.shape
            frame_fg = (label != 0).any(dim=(1, 2))  # shape: (Z,)
            if self.mode == 'train' or self.mode == 'test':
                outputs = []
                start = 0
                flag = True
                # for start in range(0, Z - window_size + 1, stride):
                while start < Z and flag:
                    end = start + window_size
                    if end > Z:
                        end = Z
                        valid_length = end - start
                        flag = False
                    else:
                        valid_length = window_size
                    fg_count = frame_fg[start:end].sum().item()
                    if fg_count >= min_fg_frames or include_background:
                        clip = image[:, start:end]                     # (C, window, H, W)
                        label_clip = label[start:end]                 # (window, H, W)
                        frame_mask = (label_clip != 0).any(dim=(1, 2)).float()
                        # 如果 T < window_size，填充至 window_size,所以在训练和测试的时候都需要针对全为0的clip做删除
                        if clip.shape[1] < window_size:
                            print('由于最后{}/{}一个帧序列不足窗口长度，因此我们需要填充空值'.format(start,end))
                            pad_size = window_size - clip.shape[1]
                            clip = F.pad(clip, (0, 0, 0, 0, 0, pad_size))  # (C, window_size, H, W)
                            label_clip = F.pad(label_clip, (0, 0, 0, 0, 0, pad_size))  # (window_size, H, W)
                            frame_mask = F.pad(frame_mask, (0, pad_size))  # (window_size,)

                        clip_frames = []
                        label_frames = []
                        for i in range(clip.shape[1]):
                            img_i = clip[:,i]
                            label_i = label_clip[i].unsqueeze(0)
                            # print(type(img_i),img_i.shape)
                            img_i = img_2D_transform(img_i)
                            label_i = label_2D_transform(label_i)
                            # 进行阈值化处理，确保标签是二值的 (0 或 1)
                            label_i = (label_i > 0.5).float()  # 使用阈值化，值大于0.5的设为1，其余设为0
                            clip_frames.append(img_i)
                            label_frames.append(label_i.squeeze())
                        #堆叠回来
                        clip_resized = torch.stack(clip_frames,dim=1)
                        label_resized = torch.stack(label_frames,dim=0)
                        outputs.append({
                            "clip": clip_resized,
                            "label": label_resized,
                            "frame_has_fg": frame_mask,
                            "start_idx": start,
                            "name":name,
                            "valid_length":valid_length
                        })
                    start += stride
                return outputs
            
            elif self.mode == 'nono':
                frame_mask = (label !=0).any(dim=(1,2)).float()
                return [{
                    "clip": image,
                    "label": label,
                    "frame_has_fg": frame_mask,
                    "start_idx": 0,
                    "name":name
                }]

    #只提取有目标的区域，此时dateloader也不用collete_fn函数
    def mask_extract_clip_slices(self, image, label, name, video_length=12,mode='test'):
        """
        根据目标区域提取2D切片：如果目标区域长度大于video_length，则随机抽取连续的video_length张2D切片；否则返回所有目标切片。

        参数:
            image (torch.Tensor): 输入图像，形状为 [C, Z, H, W]，C 为通道数，Z 为切片数，H 和 W 为图像的空间维度。
            label (torch.Tensor): 目标标签，形状为 [Z, H, W]，用于指示哪些切片含有目标。
            name (str): 输入图像的名称（例如，用于调试或记录日志）。
            video_length (int): 需要抽取的切片长度。

        返回:
            torch.Tensor: 提取的目标帧序列，形状为 [C, video_length, H, W]。
        """
        # Step 1: 找到目标区域的开始和结束索引
        # target_indices = torch.nonzero(label > 0)[:, 0].unique()  # 获取包含目标的切片索引
        # 计算每个切片中值为 1 的像素数量
        target_indices = torch.where(torch.count_nonzero(label, dim=(1, 2)) > 50)[0]  # 获取包含大于 10 个目标像素的切片索引
        print('start:{},end:{}'.format(target_indices[0],target_indices[-1]))
        
        # target_indices = []
        # for z in range(label.shape[0]):
        #     if torch.sum(label[z] == 1) > 10:  # 只选择包含大于 10 个目标像素的切片
        #         target_indices.append(z)
        # target_indices = torch.tensor(target_indices)  # 获取符合条件的切片索引
        # if len(target_indices) == 0:
        #     return torch.empty(0)  # 如果没有目标，返回空张量
        
        
        # Step 2: 识别连续的目标区域
        contiguous_segments = []
        current_segment = [target_indices[0].item()]
        
        for i in range(1, len(target_indices)):
            if target_indices[i].item() == target_indices[i - 1].item() + 1:
                # 如果当前索引与前一个索引连续，加入当前段
                current_segment.append(target_indices[i].item())
            else:
                # 如果不连续，将当前段加入 segments，并开始新的段
                contiguous_segments.append(current_segment)
                current_segment = [target_indices[i].item()]
        
        # 添加最后一个段
        contiguous_segments.append(current_segment)
        print('出现目标的连续帧序列的数量为:',len(contiguous_segments))

        #随机找一段
        '''
        random_segment = random.randint(0, len(contiguous_segments)-1)
        segment = contiguous_segments[random_segment]
        '''
        #找最大的那一段
        segment = max(contiguous_segments, key=lambda x: len(x))
        print('有肿瘤的数据长度为:{}'.format(len(segment)))
        
        #直接找开头和结尾（可能包含无肿瘤区域）
        '''
        startidx = target_indices[0]
        endidx = target_indices[-1]
        segment = [i for i in range(startidx,endidx+1)]
        '''
        # Step 3: 对每个连续的目标段进行处理
        clip_frames = []
        label_frames = []
        
        segment_length = len(segment)
        if segment_length >= video_length and mode!='test':
            # 如果该段长度大于等于 video_length，则随机抽取连续的视频长度的切片
            random_start = random.randint(0, segment_length - video_length)
            random_start = 0
            selected_indices = segment[random_start:random_start + video_length]
        else:
            # 如果该段长度小于 video_length，则返回该段所有目标切片
            selected_indices = segment
        
        # 提取目标区域的切片和标签
        selected_slices = image[:, selected_indices, :, :]
        selected_labels = label[selected_indices, :, :]
        
        # 将选中的切片和标签转换为2D
        img_2D_transform = Compose([Resize(spatial_size=self.img_size, mode='bilinear')])
        label_2D_transform = Compose([Resize(spatial_size=self.img_size, mode='bilinear')])

        for i in range(selected_slices.shape[1]):
            img_i = selected_slices[:, i]
            label_i = selected_labels[i].unsqueeze(0)
            img_i = img_2D_transform(img_i)
            label_i = label_2D_transform(label_i)
            # 进行阈值化处理，确保标签是二值的 (0 或 1)
            label_i = (label_i > 0.5).float()  # 使用阈值化，值大于0.5的设为1，其余设为0
            # print(torch.nonzero(label_i== 1))
            clip_frames.append(img_i)
            label_frames.append(label_i.squeeze())

        # 堆叠回来
        clip_resized = torch.stack(clip_frames, dim=1)
        label_resized = torch.stack(label_frames, dim=0)

        return {
            "clip": clip_resized,
            "label": label_resized,
            "start_idx": selected_indices[0],
            "name": name
        }


    def get_transforms(self):
        transforms = [
            ScaleIntensityRanged(#这是在做啥嘞,相当于起了个clip的效果，将超出intensity_range值的都截断到intensity_range范围里
                keys=["image"],
                a_min=self.intensity_range[0],
                a_max=self.intensity_range[1],
                b_min=self.intensity_range[0],
                b_max=self.intensity_range[1],
                clip=True,
            ),
        ]

        if self.mode == "train":
            transforms.extend(
                [
                    RandShiftIntensityd( # 随机强度偏移（模拟亮度变化）
                        keys=["image"],
                        offsets=20,
                        prob=0.5,
                    ),
                    CropForegroundd( # 裁剪前景（移除无效背景）
                        keys=["image", "label"],
                        source_key="image",
                        select_fn=lambda x: x > self.intensity_range[0], # 保留强度>阈值的区域
                    ),
                    NormalizeIntensityd( # 强度标准化
                        keys=["image"],
                        subtrahend=self.global_mean, # 减去全局均值
                        divisor=self.global_std, # 除以全局标准差
                    ),
                ]
            )

            # if self.do_dummy_2D:
            #     transforms.extend(  # 随机旋转（仅在XY平面）
            #         [
            #            RandRotated(
            #                 keys=["image", "label"],
            #                 prob=0.3, # 30%概率应用
            #                 range_x=30 / 180 * np.pi,   # 旋转角度范围±30度
            #                 keep_size=False, # 允许输出尺寸变化
            #                     ),
            #             RandZoomd(  # 随机缩放（各向异性，Z轴不变）
            #                 keys=["image", "label"],
            #                 prob=0.3,
            #                 min_zoom=[1, 0.9, 0.9], # Z轴缩放1倍，XY轴缩放0.9~1.1倍
            #                 max_zoom=[1, 1.1, 1.1],
            #                 mode=["trilinear", "trilinear"], # 三线性插值（保持3D连续性）
            #             ),
            #         ]
            #     )
            # else:
            #     transforms.extend(
            #         [
            #             RandZoomd(
            #                 keys=["image", "label"],
            #                 prob=0.8,
            #                 min_zoom=0.85,
            #                 max_zoom=1.25,
            #                 mode=["trilinear", "trilinear"],
            #             ),
            #         ]
            #     )

            transforms.extend(
                [
                    BinarizeLabeld(keys=["label"]), # 二值化标签（将多分类标签转为0/1）
                    # SpatialPadd(
                    #     keys=["image", "label"], # 填充图像至略大于目标尺寸
                    #     spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
                    # ),
                ])
            #         RandCropByPosNegLabeld( # 按正/负样本比例裁剪
            #             keys=["image", "label"],
            #             spatial_size=[round(i * 1.2) for i in self.rand_crop_spatial_size],
            #             label_key="label",
            #             pos=2,
            #             neg=1,
            #             num_samples=1,
            #         ),
            #         RandSpatialCropd( #最终随机裁剪到目标尺寸
            #             keys=["image", "label"],
            #             roi_size=self.rand_crop_spatial_size,
            #             random_size=False,
            #         ),
            #         # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0), #随机沿3个轴翻转
            #         # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            #         # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            #         RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3), #随机90度旋转
            #     ]
            # )
        elif (not self.do_val_crop) and (self.mode == "val"):
            transforms.extend(
                [
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif  (self.do_val_crop)  and (self.mode == "val"):
            transforms.extend(
                [
                    CropForegroundd(
                        keys=["image", "label"],
                        source_key="image",
                        select_fn=lambda x: x > self.intensity_range[0],
                    ),
                    # SpatialPadd(
                    #     keys=["image", "label"],
                    #     spatial_size=[i for i in self.rand_crop_spatial_size],
                    # ),
                    # RandCropByPosNegLabeld(
                    #     keys=["image", "label"],
                    #     spatial_size=self.rand_crop_spatial_size,
                    #     label_key="label",
                    #     pos=1,
                    #     neg=0,
                    #     num_samples=1,
                    # ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        elif self.mode == "test":
            transforms.extend(
                [
                    CropForegroundd( # 裁剪前景（移除无效背景）
                        keys=["image", "label"],
                        source_key="image",
                        select_fn=lambda x: x > self.intensity_range[0], # 保留强度>阈值的区域
                    ),
                    NormalizeIntensityd(
                        keys=["image"],
                        subtrahend=self.global_mean,
                        divisor=self.global_std,
                    ),
                    BinarizeLabeld(keys=["label"]),
                ]
            )
        else:
            raise NotImplementedError

        transforms = Compose(transforms)
            
        return transforms
    


def flatten_collate_fn(batch):
    """
    扁平化 collate_fn，用于处理 Dataset 返回 List[Dict] 的情况。
    将多个患者的 clip 列表整合成一个大的 list of dicts（可用于训练）。

    batch: List[List[Dict]] → 每个 item 是一个患者的多个 clip
    返回: List[Dict] → 所有 clip 打平后的大列表
    """
    flat_batch = []
    for clips in batch:
        flat_batch.extend(clips)  # 展平每个样本返回的 clip 列表
    return flat_batch

if __name__ == '__main__':
    colon_dataset = BaseVolumeDataset(path_prefix='G:\国防科技大学\其他对比数据集\dataset\Task10_Colon',augmentation=True)
    # datadict = colon_dataset.__getitem__(2)

    # image = datadict['img']
    # label = datadict['label'].squeeze()

    '''
    count = {}
    for i in range(50):
        print('\r进度:{}/{}'.format(i,50),end='')
        datadict = colon_dataset.__getitem__(i)
        image = datadict['img']
        label = datadict['label']
        D = image.shape[0]
        if D not in count.keys():
            count[D] = 1
        else:
            count[D] += 1
    print('\n')
    print(max(list(count.keys())))
    '''

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset=colon_dataset,batch_size=1,shuffle=True,collate_fn=flatten_collate_fn)

    for sample in loader:
        clips = torch.stack([s['clip'] for s in sample])
        labels = torch.stack([s["label"] for s in sample])
        frame_has_fg = torch.stack([s["frame_has_fg"] for s in sample])
        start_idxs = [s["start_idx"] for s in sample]
        print('暂停一下')

    
        

