# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
from threading import Thread

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_sdpa_settings():
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        # only use Flash Attention on Ampere (8.0) or newer GPUs
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn(
                "Flash Attention is disabled as it requires a GPU with Ampere (8.0) CUDA capability.",
                category=UserWarning,
                stacklevel=2,
            )
        # keep math kernel for PyTorch versions before 2.2 (Flash Attention v2 is only
        # available on PyTorch 2.2+, while Flash Attention v1 cannot handle all cases)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(
                f"You are using PyTorch {torch.__version__} without Flash Attention v2 support. "
                "Consider upgrading to PyTorch 2.2+ for Flash Attention v2 (which could be faster).",
                category=UserWarning,
                stacklevel=2,
            )
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True

    return old_gpu, use_flash_attn, math_kernel_on


def get_connected_components(mask):
    """
    Get the connected components (8-connectivity) of binary masks of shape (N, 1, H, W).

    Inputs:
    - mask: A binary mask tensor of shape (N, 1, H, W), where 1 is foreground and 0 is
            background.

    Outputs:
    - labels: A tensor of shape (N, 1, H, W) containing the connected component labels
              for foreground pixels and 0 for background pixels.
    - counts: A tensor of shape (N, 1, H, W) containing the area of the connected
              components for foreground pixels and 0 for background pixels.
    """
    from sam2_train import _C

    return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())


def mask_to_box(masks: torch.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H, W] boxes, dtype=torch.Tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)

    return bbox_coords


def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width


class AsyncVideoFrameLoader:
    """
    A list of video frames to be load asynchronously without blocking session start.
    """

    def __init__(self, img_paths, image_size, offload_video_to_cpu, img_mean, img_std):
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        # items in `self._images` will be loaded asynchronously
        self.images = [None] * len(img_paths)
        # catch and raise any exceptions in the async loading thread
        self.exception = None
        # video_height and video_width be filled when loading the first image
        self.video_height = None
        self.video_width = None

        # load the first frame to fill video_height and video_width and also
        # to cache it (since it's most likely where the user will click)
        self.__getitem__(0)

        # load the rest of frames asynchronously without blocking the session start
        def _load_frames():
            try:
                for n in tqdm(range(len(self.images)), desc="frame loading (JPEG)"):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame loading thread") from self.exception

        img = self.images[index]
        if img is not None:
            return img

        img, video_height, video_width = _load_img_as_tensor(
            self.img_paths[index], self.image_size
        )
        self.video_height = video_height
        self.video_width = video_width
        # normalize by mean and std
        img -= self.img_mean
        img /= self.img_std
        if not self.offload_video_to_cpu:
            img = img.cuda(non_blocking=True)
        self.images[index] = img
        return img

    def __len__(self):
        return len(self.images)


def load_video_frames(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """
    if isinstance(video_path, str) and os.path.isdir(video_path):
        jpg_folder = video_path
    else:
        raise NotImplementedError("Only JPEG frames are supported at this moment")

    frame_names = [
        p
        for p in os.listdir(jpg_folder)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    num_frames = len(frame_names)
    if num_frames == 0:
        raise RuntimeError(f"no images found in {jpg_folder}")
    img_paths = [os.path.join(jpg_folder, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoader(
            img_paths, image_size, offload_video_to_cpu, img_mean, img_std
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
    if not offload_video_to_cpu:
        images = images.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width

def load_video_frames_from_data(
    imgs_tensor,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """

def load_video_frames_from_data(
    imgs_tensor,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """

    num_frames = imgs_tensor.shape[0]

    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    images = imgs_tensor / 255.0

    if not offload_video_to_cpu:
        images = images.cuda()
        img_mean = img_mean.cuda()
        img_std = img_std.cuda()
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images


def fill_holes_in_mask_scores(mask, max_area):
    """
    A post processor to fill small holes in mask scores with area under `max_area`.
    """
    # Holes are those connected components in background with area <= self.max_area
    # (background regions are those with mask scores <= 0)
    assert max_area > 0, "max_area must be positive"
    labels, areas = get_connected_components(mask <= 0)
    is_hole = (labels > 0) & (areas <= max_area)
    # We fill holes with a small positive mask score (0.1) to change them to foreground.
    mask = torch.where(is_hole, 0.1, mask)
    return mask


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    #在之前的点输入上添加新的点和标签（在末尾添加）。
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}


#自己写的，关于根据显著图生成mask_prompt,point_prompt以及bbox_prompt三种提示
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

def spaced_slice_selection(confidences_dict, k, d_min):
    """
    参数：
        confidences_dict: dict[frame_id -> float]，每个frame的置信度
        k: int，要选择的帧数
        d_min: int，最小帧间隔距离
    返回：
        selected_ids: list[frame_id]，选出的帧id（如整型或字符串）
    """
    k = int(len(confidences_dict)//7+1)
    # 1. 按置信度从高到低排序 (frame_id, score)
    sorted_items = sorted(confidences_dict.items(), key=lambda x: x[1], reverse=True)

    selected_ids = []

    for frame_id, score in sorted_items:
        if all(abs(frame_id - sel_id) >= d_min for sel_id in selected_ids):
            selected_ids.append(frame_id)
        if len(selected_ids) == k:
            break

    return sorted(selected_ids)

# def spaced_slice_selection(confidences_dict, k, d_min, distance_weight=0.1):
#     """
#     加权选择帧，并考虑与已选帧之间的最小间隔。
    
#     参数：
#         confidences_dict: dict[frame_id -> float]，每个帧的置信度
#         k: int，要选择的帧数
#         d_min: int，最小帧间隔距离
#         distance_weight: float，距离对评分的权重，默认0.1
        
#     返回：
#         selected_ids: list[frame_id]，选出的帧id
#     """
#     # 1. 按置信度从高到低排序 (frame_id, score)
#     sorted_items = sorted(confidences_dict.items(), key=lambda x: x[1], reverse=True)

#     selected_ids = []  # 用来存储选择的帧

#     for frame_id, score in sorted_items:
#         # 计算每个帧的加权得分
#         # 计算距离惩罚：对于已经选中的帧，判断是否小于d_min距离
#         distance_penalty = sum(abs(frame_id - sel_id) < d_min for sel_id in selected_ids)
        
#         # 加权得分：置信度 - 距离惩罚
#         weighted_score = score - distance_weight * distance_penalty
        
#         # 如果加权得分大于0，就选中这个帧
#         if weighted_score > 0:
#             selected_ids.append(frame_id)
        
#         # 如果已选择的帧数达到k个，退出
#         if len(selected_ids) == k:
#             break

#     # 返回选中的帧id列表，按照帧id排序
#     return sorted(selected_ids)



def compute_saliency_confidence(S_map, Threshold=0.5):
#提取显著区域Ri
    Ri = S_map > Threshold
    if Ri.sum() > 0:
        saliency_values = S_map[Ri]
        confidence_score = saliency_values.mean().item()
    else:
        confidence_score = 0.0

    return confidence_score

import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def get_prob_map(sal_map, temp=1.0, normalize=True, eps=1e-6):
    """
    更稳定的 sal_map -> prob_map 转换，支持 batch-wise 归一化。
    """
    if normalize and sal_map.dim() == 4:
        mean = sal_map.mean(dim=[2,3], keepdim=True)
        std = sal_map.std(dim=[2,3], keepdim=True)
        sal_map = (sal_map - mean) / (std + eps)
    elif normalize:
        sal_map = (sal_map - sal_map.mean()) / (sal_map.std() + eps)
    else:
        print('压根没进normalize')
    prob_map = torch.sigmoid(temp * sal_map)

    return prob_map

def save_fig(save_dir,S_batch):
    os.makedirs(save_dir, exist_ok=True)  # 如果目录不存在则创建
    B, _, _, _ = S_batch.shape
    # 遍历批次中的每个样本并保存其特征图
    for i in range(B):
        sample = S_batch[i, 0, :, :].cpu().to(torch.float32).numpy()  # 提取第 i 个样本的特征图
        
        # 归一化处理（可选）
        normalized_sample = (sample - np.min(sample)) / (np.max(sample) - np.min(sample))
        
        # 设置保存路径
        save_path = os.path.join(save_dir, f"sample_{i+1}_feature_map.png")
        
        # 保存特征图
        plt.imshow(normalized_sample, cmap='gray')
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)  # 保存图像，去除多余空白
        plt.close()  # 关闭当前图像，避免内存泄漏

def generate_prompts_from_saliency_batch(S_batch, upsample_size=(512, 512), threshold=0.5, mode='train'):
    """
    从 saliency maps 批量生成 box、point、mask prompt（支持 train/eval 模式）
    参数：
        S_batch: [B, 1, H, W]，原始 saliency maps
        upsample_size: 上采样尺寸（默认为目标图大小）
        threshold: hard mask 的阈值
        mode: 'train' 或 'eval'
    返回：
        List[dict], 每帧包含 box_prompt, point_prompt, mask_prompt
    """
    B, _, _, _ = S_batch.shape

    # 创建一个文件夹来保存图像
    # save_fig(save_dir='saved_feature_maps',S_batch=S_batch.detach())

    # S_batch = F.interpolate(S_batch, size=upsample_size, mode='bilinear', align_corners=False)  # [B,1,H,W]
    # print('SAM2-SP用到了上采样，我们需要把它删了')
    prompts = []
    
    for i in range(B):
        sal_map = S_batch[i, 0]  # shape [H, W]
        H, W = sal_map.shape
        device = sal_map.device

        if mode == 'nono':
        # ------ 显著图归一化 + sigmoid + 温度调节 ------
            # sal_map = (sal_map - sal_map.mean()) / (sal_map.std() + 1e-6)
            prob_map = sal_map.clamp(-32, 32)
            # prob_map = torch.sigmoid(2.0 * sal_map)  # temp=2.0，增强区分度
            prob_map = torch.sigmoid(sal_map*5.0)
            print("prob_map mean:", prob_map.mean().item())


            # 可导 mask 用原始 sigmoid 输出（训练）
            mask_prompt = prob_map
            # ---------- soft mask ----------
            # prob_map = get_prob_map(sal_map, temp=0.5, normalize=True)
            # mask_prompt = prob_map  # 可导，训练用

            # ---------- soft box ----------
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, dtype=torch.float32, device=device),
                torch.arange(W, dtype=torch.float32, device=device),
                indexing="ij"
            )
            # print("[Debug] sal_map:", sal_map.min().item(), sal_map.max().item(), sal_map.mean().item(), sal_map.std().item())
            # print(f"[Debug] prob_map: {prob_map.min().item():.3f} ~ {prob_map.max().item():.3f}")
            total = prob_map.sum() + 1e-6
            x_mean = (x_grid * prob_map).sum() / total
            y_mean = (y_grid * prob_map).sum() / total
            x_var = ((x_grid - x_mean)**2 * prob_map).sum() / total
            y_var = ((y_grid - y_mean)**2 * prob_map).sum() / total

            # 替代 sqrt * 2，加入 soft 限制
            half_width = torch.clamp(x_var.sqrt() * 2.0, min=8.0, max=W / 2)
            half_height = torch.clamp(y_var.sqrt() * 2.0, min=8.0, max=H / 2)

            # 限制坐标范围
            x_min = torch.clamp(x_mean - half_width, min=0.0, max=W - 1.0)
            x_max = torch.clamp(x_mean + half_width, min=0.0, max=W - 1.0)
            y_min = torch.clamp(y_mean - half_height, min=0.0, max=H - 1.0)
            y_max = torch.clamp(y_mean + half_height, min=0.0, max=H - 1.0)

            box_prompt = torch.stack([x_min, y_min, x_max, y_max])

            # ---------- soft point ----------
            flat_idx = torch.argmax(prob_map)
            y_point = flat_idx // W
            x_point = flat_idx % W
            point_prompt = torch.stack([x_point, y_point]).float()

        elif mode == 'eval' or mode == 'train':
            with torch.no_grad():
                # ---------- hard mask ----------
                print("[Debug] sal_map:", sal_map.min().item(), sal_map.max().item(), sal_map.mean().item(), sal_map.std().item())
                binary_mask = (sal_map > threshold).float()
                mask_prompt = binary_mask  # 推理用，非可导

                # ---------- hard box ----------
                nonzero = binary_mask.nonzero(as_tuple=False)
                if nonzero.numel() > 0:
                    y_min, x_min = nonzero.min(dim=0).values
                    y_max, x_max = nonzero.max(dim=0).values
                    box_prompt = torch.stack([x_min, y_min, x_max, y_max]).float()
                else:
                    box_prompt = torch.tensor([0, 0, W - 1, H - 1], dtype=torch.float32, device=device)
                
                # box_prompt = torch.tensor([0, 0, W - 1, H - 1], dtype=torch.float32, device=device) #测试用的

                # ---------- hard point ----------
                binary_np = binary_mask.cpu().numpy().astype("uint8")
                dist_map = distance_transform_edt(binary_np)
                max_y, max_x = divmod(dist_map.argmax(), dist_map.shape[1])
                point_prompt = torch.tensor([max_x, max_y], dtype=torch.float32, device=device)

        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'train' or 'eval'.")

        prompts.append({
            "box_prompt": box_prompt,         # [4]
            "point_prompt": point_prompt,     # [2]
            "mask_prompt": mask_prompt        # [H, W]
        })

    return prompts

def generate_bbox(mask, variation=0, seed=None):
    if not isinstance(mask, np.ndarray):
        mask = mask.cpu().numpy()
    if seed is not None:
        np.random.seed(seed)
    # check if all masks are black
    if len(mask.shape) != 2:
        current_shape = mask.shape
        raise ValueError(f"Mask shape is not 2D, but {current_shape}")
    max_label = max(set(mask.flatten()))
    if max_label == 0:
        return np.array([np.nan, np.nan, np.nan, np.nan])
    # max agreement position
    indices = np.argwhere(mask == max_label) 
    # return point_labels, indices[np.random.randint(len(indices))]
    # print(indices)
    y0 = np.min(indices[:, 0])
    y1 = np.max(indices[:, 0])
    x0 = np.min(indices[:, 1])
    x1 = np.max(indices[:, 1])
    w = x1 - x0
    h = y1 - y0
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    if variation > 0:
        # num_rand = np.random.randn() * variation
        # w *= 1 + num_rand[0]
        # h *= 1 + num_rand[1]
        # x1 = mid_x + w / 2
        # x0 = mid_x - w / 2
        # y1 = mid_y + h / 2
        # y0 = mid_y - h / 2

        # Calculate width and height for reference
        # Generate random variations for each side (x0, y0, x1, y1)
        # variation is in [-variation, variation] * original width/height
        rand_x0 = np.random.uniform(-variation, variation) * w
        rand_y0 = np.random.uniform(-variation, variation) * h
        rand_x1 = np.random.uniform(-variation, variation) * w
        rand_y1 = np.random.uniform(-variation, variation) * h
        
        # Apply random shifts to each side
        r_x0 = x0 + rand_x0
        r_y0 = y0 + rand_y0
        r_x1 = x1 + rand_x1
        r_y1 = y1 + rand_y1
        # print('原有坐标为:',np.array([x0, y0, x1, y1]))
        return np.array([r_x0, r_y0, r_x1, r_y1])
    return np.array([x0, y0, x1, y1])


def min_max_normalize(image_tensor):
    B, C, H, W = image_tensor.shape
    # 展平所有像素（B, C*H*W）
    image_flat = image_tensor.reshape(B, -1)  # [B, C*H*W]
    # 计算 min/max，形状 [B, 1]
    min_vals = image_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)  # [B, 1, 1, 1]
    max_vals = image_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)  # [B, 1, 1, 1]
    # 归一化
    normalized = (image_tensor - min_vals) / (max_vals - min_vals + 1e-8)
    return normalized