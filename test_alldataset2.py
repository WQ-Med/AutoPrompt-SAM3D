# train.py
#!/usr/bin/env	python3

""" train network using pytorch
    Yunli Qi
"""

import os
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import cfg
# from func_3d import function_SP as function
# from func_3d import function_SP_adapter as function
# from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader
import torch.nn as nn
from sam2_train.modeling.backbones.utils import ImagePromptEncoder,MLPHead,MaskClassifier
from sam2_train.modeling.sam.mask_decoder import MaskDecoder
from sam2_train.modeling.sam.transformer import TwoWayTransformer

import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from func_3d.function_SP import eval_seg,DiceLoss,CombinedLoss,visualize_segmentation
from func_3d.dataset.datasets import ColonVolumeDataset,flatten_collate_fn,LiTSVolumeDataset
from monai.losses import DiceLoss, FocalLoss,DiceCELoss
from torch.utils.data import DataLoader
from sam2_train.utils.misc import generate_prompts_from_saliency_batch,compute_saliency_confidence,min_max_normalize
import torch.nn.functional as F
from sam2_train.utils.misc import generate_bbox,save_fig
import random
import surface_distance
from surface_distance import metrics
import medpy.metric.binary as mmb
import numpy as np
from sam2_train.Unet.unet import UNet
from ultralytics import YOLO

args = cfg.parse_args()
#定义现在的prompt是什么
# args.prompt = 'Image'
GPUdevice = torch.device('cuda', args.gpu_device)

class CLSFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        p_t = torch.exp(-bce_loss)  # p_t is the probability of the true class
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()
    
class CLSCombinedLoss(nn.Module):
    def __init__(self,bce_weight,focal_weight) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(GPUdevice))
        # self.bce_loss = nn.BCEWithLogitsLoss()
        # self.focal_loss = FocalLoss(to_onehot_y=True, gamma=2.0)

    def forward(self, inputs, targets):
        dice = self.bce_loss(inputs, targets)
        # focal = self.focal_loss(inputs, targets)
        # return self.dice_weight * dice + self.focal_weight * focal
        return dice

def spaced_slice_selection(confidences_dict, k, d_min):
    """
    参数：
        confidences_dict: dict[frame_id -> float]，每个frame的置信度
        k: int，要选择的帧数
        d_min: int，最小帧间隔距离
    返回：
        selected_ids: list[frame_id]，选出的帧id（如整型或字符串）
    """
    # 1. 按置信度从高到低排序 (frame_id, score)
    sorted_items = sorted(confidences_dict.items(), key=lambda x: x[1], reverse=True)
    k = int(len(confidences_dict)//7+1)

    selected_ids = []

    for frame_id, score in sorted_items:
        if all(abs(frame_id - sel_id) >= d_min for sel_id in selected_ids):
            selected_ids.append(frame_id)
        if len(selected_ids) == k:
            break

    return sorted(selected_ids)

def choose_prompt_id(cls_prob,k,d_min,score_threshold=0.5):
    #cls_prob是已经经过sigmoid后的cls_prob
    # Step 2: 选择概率大于 0.5 的帧
    valid_indices = torch.where(cls_prob > score_threshold)[0] # 获取所有概率大于0.5的帧索引
    # 如果没有符合条件的帧，返回空列表
    if len(valid_indices) < 2:
        return []
    # Step 3: 对符合条件的帧的概率值进行排序
    valid_probs = cls_prob[valid_indices].squeeze()  # 获取符合条件的概率值
    sorted_probs, sorted_order = torch.sort(valid_probs, descending=True)  # 排序得到概率值和对应的索引
    # 根据排序的结果获取对应的 valid_indices
    sorted_indices = valid_indices[sorted_order]

    # Step 4: 选择概率最大的 k 个帧
    selected_indices = [sorted_indices[0].item()]  # 先选择概率最大的帧
    for i in range(1, min(k, len(sorted_indices))): 
        # 遍历排序后的帧，找到符合间隔要求的帧
        for j in range(len(sorted_indices)):
            candidate_index = sorted_indices[j].item()
            # 如果当前帧的索引和已选择的帧之间的间隔大于等于 d_min，则选择此帧
            if all(abs(candidate_index - selected_index) >= d_min for selected_index in selected_indices):
                selected_indices.append(candidate_index)
                break
    return selected_indices

def confidence_filter_spaced_slice_selection(confidences_dict,threshold):
    """
    参数：
        confidences_dict: dict[frame_id -> float]，每个frame的置信度
        k: int，要选择的帧数（如果可用帧不足k，则返回所有符合条件的帧）
        threshold: float，置信度阈值
    返回：
        selected_ids: list[frame_id]，按置信度降序排列的帧id
    """
    k = int(len(confidences_dict)//7+1)
    # 1. 过滤并降序排序
    filtered_items = [
        (fid, score) 
        for fid, score in confidences_dict.items() 
        if score >= threshold
    ]
    filtered_items.sort(key=lambda x: x[1], reverse=True)  # 按score降序
    
    # 2. 处理无符合条件帧的情况
    if not filtered_items:
        return []
    # 3. 安全获取前k个（即使不足k个）
    selected_ids = [fid for fid, _ in filtered_items[:min(k, len(filtered_items))]]
    
    return selected_ids 

def find_confidence_bounds(classification_label, threshold=0.5,min_frame_threshold = 5):
    """
    找到 Tensor 中所有 > threshold 的元素的最小覆盖范围 [start_idx, end_idx]
    
    参数:
        classification_label (torch.Tensor): 1D 概率值 Tensor（CUDA 或 CPU）
        threshold (float): 置信度阈值（默认 0.5）
    
    返回:
        tuple: (start_idx, end_idx) 或 None（如果没有满足条件的元素）
    """
    # 确保输入是 1D Tensor
    assert classification_label.dim() == 1, "Input must be a 1D tensor"
    
    # 获取大于阈值的布尔掩码
    mask = classification_label > threshold
    
    # 如果没有满足条件的元素，返回 None
    if not mask.any():
        return None
    
    # 找到所有满足条件的索引
    indices = torch.nonzero(mask, as_tuple=True)[0]
    
    # 返回最小和最大索引
    min_indice, max_indice = indices.min().item(), indices.max().item()
    if max_indice - min_indice + 1 < min_frame_threshold:
        return None
    return (indices.min().item(), indices.max().item())

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True, method = 'AFP',yolo_model = None):
     # eval mode
    net.eval()

    n_val = len(val_loader)  # the number of batch
    all_temp = [0,0]
    all_tot = 0
    all_hd95 = 0
    mix_res = (0,)*1*2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    windows_size = 12 #数据采集时滑动窗口的长度

    lossfunc = DiceLoss(include_background=False,sigmoid=True,reduction='none')

    prompt = args.prompt
    with torch.no_grad():
        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for pack in val_loader:
                sing_image_temp = [0,0]
                sing_image_hd95 = 0
                sing_image_nsd = 0
                if len(pack) == 0:
                    continue
                torch.cuda.empty_cache()
                clips = torch.stack([s['clip'] for s in pack]) #(T,C,Z,H,W)
                labels = torch.stack([s["label"] for s in pack]) #(T,Z,H,W)
                frame_has_fg = torch.stack([s["frame_has_fg"] for s in pack])  #(T,Z)
                start_idxs = [s["start_idx"] for s in pack] #一维列表
                name = [s["name"] for s in pack] #一维列表，元素相同
                valid_lengths = [s['valid_length'] for s in pack]
                # spacing = torch.stack([s['img_spacing'] for s in pack])
                T_length = clips.shape[0]
                Ignore = 0
                for t in range(T_length):
                    valid_length = valid_lengths[t]
                    imgs_tensor = clips[t].permute(1,0,2,3)
                    mask_dict = labels[t].to(dtype = torch.float32, device = GPUdevice)
                    start_idx = start_idxs[t]
                    name = name[0]
                    if valid_length < windows_size:
                        imgs_tensor = imgs_tensor[:valid_length]
                        mask_dict = mask_dict[:valid_length]
                    video_length = mask_dict.shape[0]

                    imgs_tensor = imgs_tensor.to(dtype = torch.float32, device = GPUdevice)

                    train_state = net.train_init_state(imgs_tensor=imgs_tensor)
                    features = net.forward_image(imgs_tensor)
                    # ---------------------------------------------用Unet解码器来生成伪mask提示---------------------
                    feat32 = features['vision_features']
                    feat64 = features['backbone_fpn'][-2]
                    feat128 = features['backbone_fpn'][-3]
                    if video_length != 1:
                        classification_label = net.cls_head(feat32).squeeze()
                    salient_region = find_confidence_bounds(classification_label,threshold=0.5,min_frame_threshold=5)
                    if salient_region is None:
                        Ignore += 1
                        # mask_dict = mask_dict.unsqueeze(1)
                        # temp = eval_seg(torch.zeros_like(mask_dict).to(device=GPUdevice),mask_dict,threshold)
                        # print(temp[0],temp[1])
                        # sing_image_temp[0] += temp[0]
                        # sing_image_temp[1] += temp[1]
                        continue
                    start,end = salient_region
                    print('开始/结束:{}/{}'.format(start,end))
                    if method == 'AFP':
                        disguise_mask = net.SaliencyMLP(feat128, feat64, feat32)
                    if method == 'SAM2-SP':
                        disguise_mask = net.SaliencyMLP(feat32)
                    if method == 'UNet':
                        disguise_mask,_,_,_ = net.SaliencyMLP(imgs_tensor)
                    if method in ['AFP','SAM2-FP','Unet']:
                        if any(name == "confidence_filter" for name, _ in net.named_modules()):
                            persudo_mask = torch.where(disguise_mask > 0.5, 
                                torch.tensor(255).to(dtype = torch.float32, device = GPUdevice), 
                                torch.tensor(0).to(dtype = torch.float32, device = GPUdevice))
                            logits = net.confidence_filter(persudo_mask) 
                            all_confidences = {frame_id:logits for frame_id,logits in enumerate(logits)}
                            all_confidences = {k: v for k, v in all_confidences.items() if start <= k <= end}
                            prompt_frame_id = confidence_filter_spaced_slice_selection(all_confidences,threshold=0.5)
                            if len(prompt_frame_id) == 0:
                                #SAM2-SP的显著图选帧方法
                                all_confidences = {
                                    frame_id: compute_saliency_confidence(Si.squeeze())
                                    for frame_id, Si in enumerate(disguise_mask.detach())
                                    }
                                prompt_frame_id = spaced_slice_selection(all_confidences,k=3,d_min=4)
                        else:
                            #SAM2-SP的显著图选帧方法
                            all_confidences = {
                                frame_id: compute_saliency_confidence(Si.squeeze())
                                for frame_id, Si in enumerate(disguise_mask.detach(),start=start) 
                                if start <= frame_id <= end
                                }
                            prompt_frame_id = spaced_slice_selection(all_confidences,k=3,d_min=4)
                            print('置信帧的选择为:',prompt_frame_id)
                        
                        #选择策略
                        '''
                        #选择所有帧
                        prompt_frame_id = [i for i in range(video_length)]
                        #随机选择帧
                        # prompt_frame_id = random.sample(prompt_frame_id, k=3)
                        '''
                        
                    
                        id_to_prompt_idx = {s:i for i,s in enumerate(prompt_frame_id)}
                        prompts = generate_prompts_from_saliency_batch(disguise_mask[prompt_frame_id].detach())
                    elif method == 'yolo':
                        imgs_tensor = min_max_normalize(image_tensor=imgs_tensor.detach())
                        results = yolo_model.predict(imgs_tensor, imgsz=512, conf=0)
                        confs = [] #置信度
                        for i in range(len(results)):
                            confs.append(results[i].boxes.conf[0].cpu().numpy())
                        k = len(confs) // 7 + 1
                        prompt_frame_id = []
                        if len(confs) == 0:
                            prompt_frame_id = [0]
                        else:
                            prompt_frame_id = np.argsort(confs)[::-1][:min(k, len(confs))].tolist()

                        print('置信帧的选择为:',prompt_frame_id)
                    # ---------------------------------------------用Unet解码器来生成伪mask提示---------------------
                    obj_list = [1.0]
                    for id in prompt_frame_id:
                        for ann_obj_id in obj_list:
                            try:
                                if prompt == 'click':
                                    points = prompts['point_prompt']
                                    # labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                    _, _, _ = net.train_add_new_points( #它们都说这是第一帧的处理方式,我可以试一下看看是不是第一帧的处理方式
                                        inference_state=train_state, #我感觉它更像是一种针对于有帧来进行处理的一个东西
                                        frame_idx=id,
                                        obj_id=ann_obj_id,
                                        points=points,
                                        labels=mask_dict,
                                        clear_old_points=False,
                                    )
                                elif prompt == 'bbox' or prompt == 'Autobbox' or prompt == 'Auto':
                                    if method == 'yolo':
                                        bbox = results[id].boxes[0].xyxy
                                    else:
                                        bbox = prompts[id_to_prompt_idx[id]]['box_prompt']
                                    label_bbox = generate_bbox(mask=mask_dict[id].squeeze())
                                    print('当前bbox为:',bbox,label_bbox)
                                    if not bbox == None:
                                        _, _, _ = net.train_add_new_bbox(
                                            inference_state=train_state,
                                            frame_idx=id,
                                            obj_id=ann_obj_id,
                                            bbox=torch.tensor(bbox, dtype=torch.float32).to(GPUdevice),
                                            clear_old_points=False,
                                        )
                                    else:
                                        _, _, _ = net.train_add_new_mask(
                                        inference_state=train_state,
                                        frame_idx=id,
                                        obj_id=ann_obj_id,
                                        mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                                        )
                            except KeyError:
                                _, _, _ = net.train_add_new_mask(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                                )
                    video_segments = {}  # video_segments contains the per-frame segmentation results
                
                    for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=start,max_frame_num_to_track=end-start):
                        video_segments[out_frame_idx] = {
                            out_obj_id: out_mask_logits[i]
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }

                    loss = 0
                    pred_iou = 0
                    pred_dice = 0
                    pred_hd95 = 0
                    need_to_see_flag = False
                    for id in range(video_length):
                        for ann_obj_id in obj_list:
                            if  start <= id <= end:
                                pred = video_segments[id][ann_obj_id]
                                pred = pred.unsqueeze(0)
                            else:
                                pred = torch.zeros((1, 1, 512, 512), 
                                                    dtype = torch.float32,
                                                    device=GPUdevice)
                            # print(pred,pred.shape)
                            # pred = torch.sigmoid(pred)
                            try:
                                mask = mask_dict[id].squeeze().to(dtype = torch.float32, device = GPUdevice).view(1,1,512,512)
                            except KeyError:
                                mask = torch.zeros_like(pred).to(device=GPUdevice)
                            '''
                            if args.vis:
                                os.makedirs(f'./temp/val/{name[0]}/{id}', exist_ok=True)
                                fig, ax = plt.subplots(1, 3)
                                ax[0].imshow(imgs_tensor[id, :, :, :].cpu().permute(1, 2, 0).numpy().astype(int))
                                ax[0].axis('off')
                                ax[1].imshow(pred[0, 0, :, :].cpu().numpy() > 0.5, cmap='gray')
                                ax[1].axis('off')
                                ax[2].imshow(mask[0, 0, :, :].cpu().numpy(), cmap='gray')
                                ax[2].axis('off')
                                plt.savefig(f'./temp/val/{name[0]}/{id}/{ann_obj_id}.png', bbox_inches='tight', pad_inches=0)
                                plt.close()
                            '''
                            loss += lossfunc(pred, mask)
                            temp = eval_seg(pred, mask, threshold)
                            pred = pred.cpu().numpy()
                            mask = mask.cpu().numpy()
                            pred = (pred > 0.5).astype(np.uint8)
                            if np.all(pred==0) and np.all(mask==0):
                                pred_hd95 += 0
                            elif np.all(pred==0) or np.all(mask==0):
                                pred_hd95 += 100
                            else:
                                pred_hd95 += mmb.hd95(pred,mask)
                            print(temp[0],temp[1])
                            # if temp[1] < 0.73:
                            #     print('有问题啊哥们:',name)
                            pred_iou += temp[0]
                            pred_dice += temp[1]
                            # pred_dice += 1 - DiceLoss
                            if args.vis:
                                is_choose = id in prompt_frame_id
                                if temp[1]<0.7 and need_to_see_flag == False:
                                    name[0] = name[0] + 'need_to_see'
                                    need_to_see_flag = True
                                save_path = f'./temp/IOU/SAM-SP-Unet/{name[0]}/{id}_{temp[1]}.png'
                                base_dir = "./temp/IOU/SAM-SP-Unet"
                                name_dir = name[0]

                                # 检查 name[0] 的目录是否存在
                                name_path = os.path.join(base_dir, name_dir)
                                if not os.path.exists(name_path):
                                    print(f"Directory '{name_dir}' does not exist. Creating...")
                                    os.makedirs(name_path, exist_ok=True)
                                prompt_mask = train_state['prompt_all_s'][id]
                                prompts = net.generate_prompt_auto(train_state,id,mode='eval')[0]
                                bbox = prompts['box_prompt']
                                # 继续创建子目录
                                visualize_segmentation(imgs_tensor[id], pred, mask,prompt_mask,bbox,save_path, is_choose,temp)
                    total_num = video_length
                    temp = (pred_iou/ total_num, pred_dice / total_num)
                    temp_hd95 = pred_hd95/ total_num
                    sing_image_temp[0] += temp[0]
                    sing_image_temp[1] += temp[1]
                    sing_image_hd95 += temp_hd95
                    net.reset_state(train_state)
                if T_length == Ignore:
                    all_temp[0] += 1
                    all_temp[1] += 1
                    all_hd95 += 0
                else:
                    all_temp[0] += sing_image_temp[0] / (T_length-Ignore)
                    all_temp[1] += sing_image_temp[1] / (T_length-Ignore)
                    all_hd95 += sing_image_hd95 / (T_length - Ignore)
                pbar.update()
                print(all_temp,all_hd95)    
    return all_tot/ n_val , tuple([a/n_val for a in all_temp]), all_hd95/ n_val


def main():
    torch.multiprocessing.set_start_method('spawn')
    #load_dict
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    net.cls_head = MLPHead(input_dim=256*32*32,hidden_dim = 256,output_dim=1) #这个待会去计算一下它怎么输出的
    checkpoint = torch.load('checkpoints/service_model/LiTS17/LiT_二分类模型_0.84_0.8_0.94_0.85/Model/latest_epoch.pth')
    model_weights = checkpoint['model']
    net.cls_head.load_state_dict(model_weights)

    # confidence_filter = MaskClassifier()
    # confidence_filter.load_state_dict(torch.load('checkpoints/service_model/置信帧选择模型split.pkl0.81-1.0-0.625--0.749/Model/best_epoch95.pth')['model'])
    # net.confidence_filter = confidence_filter

    #用Unet来做自动提示生成器
    # Unet提示生成器
    # unet_generator = UNet(n_channels=3,n_classes=1)
    # unet_weights = torch.load('checkpoints/service_model/Unet_train_0.26-0.36/Model/model_emb.best_epoch.pth')['model']
    # unet_generator.load_state_dict(unet_weights)
    # net.SaliencyMLP = unet_generator
    # yolo = YOLO('checkpoints/service_model/Yolo/colon.pt')

    #用yolo来做自动提示生成器



    net.to(dtype=torch.bfloat16)
    net.to(device=GPUdevice)
    # print(net)
    
    print('暂停一下')

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    # logger.info(f'我将stots打开了vartion，并设置为0.6，为了判断粗框下的指标质量')
    #数据集构造
    # test_dataset = ColonVolumeDataset(path_prefix='data/Task10_Colon',augmentation=False,mode='test',choose='all')
    test_dataset = LiTSVolumeDataset(path_prefix='data/Task03_LITS17',augmentation=False,mode='test')

    nice_test_loader = DataLoader(test_dataset,collate_fn=flatten_collate_fn)

    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    best_dice = 0.0
    best_epoch = 0
    
    for epoch in range(settings.EPOCH):
        #冻结所有参数，只做测试
        for param in net.parameters():
            param.requires_grad = False
        
        tol , (eiou , edice), hd95 = validation_sam(args,nice_test_loader, epoch, net, writer,method='AFP',yolo_model=None)
        logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice}, HD95: {hd95} || @ epoch {epoch}.')
        
        time.sleep(0.3)

    writer.close()


if __name__ == '__main__':
    main()