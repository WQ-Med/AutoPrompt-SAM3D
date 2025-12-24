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
from sam2_train.modeling.backbones.utils import ImagePromptEncoder,ClSHead,MaskClassifier,Confidence_MLP
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
import medpy.metric.binary as mmb
import numpy as np
from sam2_train.Unet.unet import UNet
import time
args = cfg.parse_args()
from ultralytics import YOLO
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

def choose_prompt_id(cls_prob,k,d_min):
    #cls_prob是已经经过sigmoid后的cls_prob
    # Step 2: 选择概率大于 0.5 的帧
    valid_indices = torch.where(cls_prob > 0.5)[0] # 获取所有概率大于0.5的帧索引
    # 如果没有符合条件的帧，返回空列表
    if len(valid_indices) == 0:
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

def train_sam(args, net: nn.Module, encoder_pot,prompt_opt,other_opt,component_opt,train_loader,
          epoch,encoder_scheduler,prompt_scheduler,other_scheduler,componet_scheduler):
    Cls_loss = 0
    Disguise_loss = 0
    epoch_loss = 0
    has_fig_count = 0 #记录有多少次是有提示帧出现的轮
    mix_res = (0,)*1*2
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    # train mode
    net.train()
    video_length = args.video_length

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    prompt = args.prompt

    # lossfunc = criterion_G
    # S_lossfunc = DiceLoss(sigmoid=True)
    # S_lossfunc =torch.nn.BCEWithLogitsLoss()
    # S_lossfunc = CombinedLoss(dice_weight=1/2, focal_weight=1 / 2)
    S_lossfunc = DiceCELoss(include_background=False, to_onehot_y=True, sigmoid=True)
    # S_lossfunc =torch.nn.BCEWithLogitsLoss()
    # lossfunc = CombinedLoss(dice_weight=1/2, focal_weight=1 / 2)
    lossfunc = DiceCELoss(include_background=False, to_onehot_y=True, sigmoid=True)
    # lossfunc =  DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5, lambda_ce=0.5)
    cls_lossfunc = CLSFocalLoss()

    seed = torch.randint(1,11,(1,7))

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            #定义一个pack的loss
            if len(pack) == 0:
                continue
            pack_cls_loss, pack_disguise_loss, pack_prompt_loss, pack_nonprompt_loss = 0, 0, 0, 0

            torch.cuda.empty_cache()
            imgs_tensor = pack['clip'].permute(0,2,1,3,4)
            mask_dict  = pack['label'].permute(1,0,2,3).to(dtype = torch.float32, device = GPUdevice)
            start_idx = pack['start_idx']
            name = pack['name']
            video_length = mask_dict.shape[0]

            imgs_tensor = imgs_tensor.squeeze(0)
            imgs_tensor = imgs_tensor.to(dtype = torch.float32, device = GPUdevice)
            
            train_state = net.train_init_state(imgs_tensor=imgs_tensor)
            features = net.forward_image(imgs_tensor)
            # ---------------------------------------------用Unet解码器来生成伪mask提示---------------------
            has_fig_count += 1
            feat32 = features['vision_features'] #(B,256,32,32)
            feat64 = features['backbone_fpn'][-2] #(B,64,64,64)
            feat128 = features['backbone_fpn'][-3] #(B,32,128,128)
            
            disguise_mask = net.SaliencyMLP(feat128, feat64, feat32)
            all_confidences = {
                frame_id: compute_saliency_confidence(Si.squeeze())
                for frame_id, Si in enumerate(disguise_mask.detach())
                }
            prompt_frame_id = spaced_slice_selection(all_confidences,k=3,d_min=4)
            print('置信帧的选择为:',prompt_frame_id)
            id_to_prompt_idx = {s:i for i,s in enumerate(prompt_frame_id)}
            # print(disguise_mask.shape)
            prompts = generate_prompts_from_saliency_batch(disguise_mask[prompt_frame_id].detach())
            #计算伪mask损失
            # disguise_loss = S_lossfunc(disguise_mask,F.interpolate(mask_dict, size=disguise_mask.shape[-2:], mode='bilinear', align_corners=False)) #(4,1,H,W)
            disguise_loss = S_lossfunc(disguise_mask,mask_dict)
            # if epoch == 1:
            #     print('检查')
            if epoch < 100:
                for id in range(len(prompts)):
                    bbox = prompts[id]['box_prompt']
                    label_bbox = generate_bbox(mask=mask_dict[prompt_frame_id[id]].squeeze())
                    # print('当前bbox为:',bbox,label_bbox)
                for p in net.SaliencyMLP.parameters():
                    if p.requires_grad:
                        continue
                    else:
                        print('SaliencyMLP确实没有解冻')
                # for p in net.image_encoder.parameters():
                #     if p.requires_grad:
                #         continue
                #     else:
                #         print('image_encoder确实没有解冻')
                encoder_pot.zero_grad()
                prompt_opt.zero_grad()
                other_opt.zero_grad()
                component_opt.zero_grad()
                disguise_loss.backward()
                for name, param in net.SaliencyMLP.named_parameters():
                    if param.grad is not None:
                        print(f"{name} 的梯度均值: {param.grad.mean().item()}")
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                encoder_pot.step()
                prompt_opt.step()
                other_opt.step()
                component_opt.step()
                pbar.set_postfix(**{'loss (batch)': disguise_loss.item()})
                pbar.update()
                continue
            pack_disguise_loss += disguise_loss.item()
            # ---------------------------------------------用Unet解码器来生成伪mask提示---------------------
            # prompt_frame_id = torch.tensor(prompt_frame_id)
            id_to_prompt_idx = {s:i for i,s in enumerate(prompt_frame_id)}
            prompts = generate_prompts_from_saliency_batch(disguise_mask[prompt_frame_id])
            # prompt_frame_id = list(range(0, video_length, prompt_freq))
            obj_list = [1.0]

            # reverse = np.random.rand() > 0.5

            
            with torch.cuda.amp.autocast():
                #去掉prompt_encoder
                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        try:
                            if prompt == 'click':
                                points = prompts[id_to_prompt_idx[id]]['point_prompt']
                                # labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.train_add_new_points( #它们都说这是第一帧的处理方式,我可以试一下看看是不是第一帧的处理方式
                                    inference_state=train_state, #我感觉它更像是一种针对于有帧来进行处理的一个东西
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=mask_dict,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox' or prompt == 'Autobbox':
                                bbox = prompts[id_to_prompt_idx[id]]['box_prompt']
                                label_bbox = generate_bbox(mask=mask_dict[id].squeeze())
                                # print('当前bbox为:',bbox,label_bbox)
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
            
                for out_frame_idx, out_obj_ids, out_mask_logits in net.train_propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                #计算最终的loss值
                loss = 0
                non_prompt_loss = 0
                prompt_loss = 0
                pred_iou = 0
                pred_dice = 0
                for id in range(video_length):
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.view(1,1,512,512)
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id].to(dtype = torch.float32, device = GPUdevice).view(1,1,512,512)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)
                        if args.train_vis:
                            os.makedirs(f'./temp/train/{name[0]}/{id}', exist_ok=True)
                            fig, ax = plt.subplots(1, 3)
                            ax[0].imshow(imgs_tensor[id, :, :, :].detach().cpu().permute(1, 2, 0).numpy().astype(int))
                            ax[0].axis('off')
                            ax[1].imshow(pred[0, 0, :, :].detach().cpu().numpy() > 0.5, cmap='gray')
                            ax[1].axis('off')
                            try:
                                # bbox = bbox_dict[id][ann_obj_id]
                                ax[1].add_patch(plt.Rectangle((bbox[0][0], bbox[0][1]), bbox[0][2] - bbox[0][0], bbox[0][3] - bbox[0][1], edgecolor='green', facecolor=(0,0,0,0), lw=2))
                            except KeyError:
                                pass
                            ax[2].imshow(mask[0, 0, :, :].detach().cpu().numpy(), cmap='gray')
                            ax[2].axis('off')
                            plt.savefig(f'./temp/train/{name[0]}/{id}/{obj_list.index(ann_obj_id)}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()
                        obj_loss = lossfunc(pred, mask)
                        temp = eval_seg(pred, mask, threshold)
                        pred_iou += temp[0]
                        # if pred_iou > 0.2:
                        #     print('看一看')
                        pred_dice += temp[1]
                        # print(pred,mask)
                        loss += obj_loss.item()
                        if id in prompt_frame_id:
                            prompt_loss += obj_loss
                        else:
                            non_prompt_loss += obj_loss

                loss = loss / video_length / len(obj_list)
                non_prompt_loss = non_prompt_loss / (video_length - len(prompt_frame_id)) / len(obj_list)
                prompt_loss = prompt_loss / len(prompt_frame_id) / len(obj_list)
                pack_nonprompt_loss += non_prompt_loss.item()
                if isinstance(prompt_loss, float):
                    print('暂停一下')
                pack_prompt_loss += prompt_loss.item()
                all_loss = disguise_loss + prompt_loss + non_prompt_loss
                encoder_pot.zero_grad()
                prompt_opt.zero_grad()
                other_opt.zero_grad()
                component_opt.zero_grad()
                all_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                encoder_pot.step()
                prompt_opt.step()
                other_opt.step()
                component_opt.step()

                pbar.set_postfix(**{'loss (batch)': loss})
                pbar.update()

            total_num = video_length * len(obj_list)
            temp = (pred_iou / total_num, pred_dice / total_num)
            mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
            print('当前{}/{}的clip指标为{},{}'.format(name,start_idx,temp[0],temp[1]))
            net.reset_state(train_state)

        encoder_scheduler.step()
        prompt_scheduler.step()
        other_scheduler.step()
        componet_scheduler.step()
        
    return epoch_loss / len(train_loader),0,0

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True,method='yolo',yolo_model = None):
     # eval mode
        #运行时间测试
    
    net.eval()

    n_val = len(val_loader)  # the number of batch
    mix_res = (0,)*1*2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    all_hd95 = 0
    prompt = args.prompt
    # lossfunc = criterion_G
    lossfunc = DiceLoss(include_background=False,sigmoid=True,reduction='none')
    # lossfunc = paper_loss
    with torch.no_grad():
        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for pack in val_loader:
                if len(pack) == 0:
                    continue
                # start_time = time.time()
                torch.cuda.empty_cache()
                imgs_tensor = pack['clip'].permute(0,2,1,3,4)
                mask_dict = pack['label'].permute(1,0,2,3).to(dtype = torch.float32, device = GPUdevice)
                start_idx = pack['start_idx']
                video_length = mask_dict.shape[0]
                name = pack['name']

                imgs_tensor = imgs_tensor.squeeze(0)
                imgs_tensor = imgs_tensor.to(dtype = torch.float32, device = GPUdevice)

                train_state = net.train_init_state(imgs_tensor=imgs_tensor)
                features = net.forward_image(imgs_tensor)
                # ---------------------------------------------用Unet解码器来生成伪mask提示---------------------
                feat32 = features['vision_features']
                feat64 = features['backbone_fpn'][-2]
                feat128 = features['backbone_fpn'][-3]
                if method == 'AFP':
                    disguise_mask = net.SaliencyMLP(feat128, feat64, feat32)
                elif method == 'SAM2-SP':
                    disguise_mask = net.SaliencyMLP(feat32)
                elif method == 'UNet':
                    disguise_mask,_,_,_ = net.SaliencyMLP(imgs_tensor)
                if method in ['AFP','UNet','SAM2-SP']:
                    if any(name == "confidence_filter" for name, _ in net.named_modules()):
                        persudo_mask = torch.where(disguise_mask > 0.5, 
                            torch.tensor(255).to(dtype = torch.float32, device = GPUdevice), 
                            torch.tensor(0).to(dtype = torch.float32, device = GPUdevice))
                        logits = net.confidence_filter(persudo_mask)
                        all_confidences = {frame_id:logits for frame_id,logits in enumerate(logits)}
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
                            for frame_id, Si in enumerate(disguise_mask.detach())
                            }
                        prompt_frame_id = spaced_slice_selection(all_confidences,k=3,d_min=4)

                    #选择策略
                    '''
                    #选择所有帧
                    prompt_frame_id = [i for i in range(video_length)]
                    # 随机选择帧
                    prompt_frame_id = random.sample(prompt_frame_id, k=3)
                    '''

                    print('置信帧的选择为:',prompt_frame_id)
                    id_to_prompt_idx = {s:i for i,s in enumerate(prompt_frame_id)}
                    prompts = generate_prompts_from_saliency_batch(disguise_mask[prompt_frame_id].detach())

                elif method == 'yolo':
                    imgs_tensor = min_max_normalize(image_tensor=imgs_tensor.detach())
                    results = yolo_model.predict(imgs_tensor, imgsz=512, conf=0)
                    disguise_mask = torch.zeros_like(imgs_tensor)
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
                                _, _, _ = net.train_add_new_points( 
                                    inference_state=train_state, 
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
                                    # bbox = [124,232,400,400]
                                label_bbox = generate_bbox(mask=mask_dict[id].squeeze())
                                # print('当前bbox为:',bbox,label_bbox)
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
            
                for out_frame_idx, out_obj_ids, out_mask_logits in net.propagate_in_video(train_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: out_mask_logits[i]
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                # end_time = time.time()
                # print("耗时: {:.2f}秒".format(end_time - start_time))
                # raise ValueError()

                imgs_tensor = min_max_normalize(image_tensor=imgs_tensor.detach()) #切记切记啊一定要切记啊，这只是为了画图而已
                loss = 0
                pred_iou = 0
                pred_dice = 0
                pred_hd95 = 0
                need_to_see_flag = False
                for id in range(video_length):
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        # print(pred,pred.shape)
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id].squeeze().to(dtype = torch.float32, device = GPUdevice).view(1,1,512,512)
                        except KeyError:
                            mask = torch.zeros_like(pred).to(device=GPUdevice)

                        loss += lossfunc(pred, mask)
                        temp = eval_seg(pred, mask, threshold)
                        print(temp[0],temp[1])
                        pred = pred.cpu().numpy()
                        mask = mask.cpu().numpy()
                        pred = (pred > 0.5).astype(np.uint8)
                        if np.all(pred==0) and np.all(mask==0):
                            hd95 = 0
                        elif np.all(pred==0) or np.all(mask==0):
                            hd95 = 1
                        else:
                            hd95 = mmb.hd95(pred,mask)
                        if args.vis:
                            is_choose = id in prompt_frame_id
                            if temp[1]<1: #h看着改就行
                                save_path = f'./temp/new_Colon/{method}_Saliency_map/{name[0]}/{id}_{temp[1]}.png'
                                base_dir = f"./temp/new_Colon/{method}_Saliency_map/"
                                name_dir = name[0]

                                # 检查 name[0] 的目录是否存在
                                name_path = os.path.join(base_dir, name_dir)
                                if not os.path.exists(name_path):
                                    print(f"Directory '{name_dir}' does not exist. Creating...")
                                    os.makedirs(name_path, exist_ok=True)
                                # 继续创建子目录
                                visualize_segmentation(imgs_tensor[id], pred, mask,disguise_mask[id].unsqueeze(0).float(),generate_bbox(mask=disguise_mask[id].unsqueeze(0).float().squeeze().cpu().numpy()),save_path, is_choose,temp,id,generate_bbox(mask=mask.squeeze()))
                                # visualize_segmentation(imgs_tensor[id], pred, mask,disguise_mask[id].unsqueeze(0).float(),results[id].boxes[0].xyxy[0].cpu(),save_path, is_choose,temp,id,generate_bbox(mask=mask.squeeze()))

                        # if temp[1] < 0.73:
                        #     print('有问题啊哥们:',name)
                        pred_iou += temp[0]
                        pred_dice += temp[1]
                        pred_hd95 += hd95
                        # pred_dice += 1 - DiceLoss


                total_num = video_length * len(obj_list)
                loss = loss / total_num
                # print('loss',loss)
                temp = (pred_iou / total_num, pred_dice / total_num)
                pred_hd95 = pred_hd95 / total_num
                tot += loss

                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
                all_hd95 += pred_hd95

                net.reset_state(train_state)
                pbar.update()

    return tot/ n_val , tuple([a/n_val for a in mix_res]) , all_hd95/n_val


def main():
    torch.multiprocessing.set_start_method('spawn')
    #load_dict
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    #如果加载已经换好了的模型权重
    if args.pretrain:
        print(args.pretrain)
        weights = torch.load(args.pretrain)
        net.load_state_dict(weights,strict=False)
    else:
        #设置一个二分类器
        # net.cls_head = ClSHead(hidden_dim=256) #这个待会去计算一下它怎么输出的
        # confidence_filter = MaskClassifier()
        # confidence_filter.load_state_dict(torch.load('checkpoints/service_model/Colon/置信帧选择模型split.pkl0.81-1.0-0.625--0.749/Model/best_epoch95.pth')['model'])
        # net.confidence_filter = confidence_filter

        # Unet提示生成器
        # unet_generator = UNet(n_channels=3,n_classes=1)
        # unet_weights = torch.load('checkpoints/service_model/Unet_train_0.26-0.36/Model/model_emb.best_epoch.pth')['model']
        # unet_generator.load_state_dict(unet_weights)
        # net.SaliencyMLP = unet_generator

        # YOLO提示生成器
        # yolo = YOLO('checkpoints/service_model/Colon/YOLO/weights/best.pt')

        net.to(dtype=torch.bfloat16)
        net.to(device=GPUdevice)
    # print(net)
    
    print('暂停一下')


    start_epoch = 0
    # 多个optimizer
    other_layer = (
                  []
                  + list(net.obj_ptr_proj.parameters())
                  + list(net.memory_encoder.parameters())
                  + list(net.memory_attention.parameters())
                  + list(net.mask_downsample.parameters())
    )
    new_component_layer = (
        []
        + list(net.SaliencyMLP.parameters())
        # + list(net.cls_head.parameters())
    )
    prompt_layer = (
        []
        + list(net.sam_prompt_encoder.parameters())
        + list(net.sam_mask_decoder.parameters())
    )
    encoder_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.image_encoder.parameters()),lr=1e-4,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    prompt_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, prompt_layer),lr=1e-4,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    other_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, other_layer),lr=1e-8,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=1e-4,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # component_opt= torch.optim.RMSprop(filter(lambda p: p.requires_grad, new_component_layer),lr=5e-5,eps=1e-08, weight_decay=0)
    component_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, new_component_layer),lr=1e-4,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    
    encoder_scheduler = torch.optim.lr_scheduler.LinearLR(encoder_opt, start_factor=1.0 - start_epoch * (1.0 - 0.01) / args.max_epoch, end_factor=0.01, total_iters=args.max_epoch)
    prompt_scheduler = torch.optim.lr_scheduler.LinearLR(prompt_opt,start_factor=1.0 - start_epoch * (1.0 - 0.01) / args.max_epoch, end_factor=0.01, total_iters=args.max_epoch)
    other_scheduler = torch.optim.lr_scheduler.LinearLR(other_opt,start_factor=1.0 - start_epoch * (1.0 - 0.01) / args.max_epoch, end_factor=0.01, total_iters=args.max_epoch)
    component_scheduler = torch.optim.lr_scheduler.LinearLR(component_opt,start_factor=1.0 - start_epoch * (1.0 - 0.01) / args.max_epoch, end_factor=0.01, total_iters=args.max_epoch)

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.path_helper = set_log_dir('logs', args.exp_name)
    logger = create_logger(args.path_helper['log_path'])
    logger.info(args)

    # logger.info(f'我将stots打开了vartion，并设置为0.6，为了判断粗框下的指标质量')
    #colon数据集构造
    '''
    train_dataset = ColonVolumeDataset(path_prefix='data/Task10_Colon',augmentation=True,choose='mask')
    val_dataset = ColonVolumeDataset(path_prefix='data/Task10_Colon',augmentation=False,mode='val',rand_crop_spatial_size=(128,512,512))
    '''
    test_dataset = ColonVolumeDataset(path_prefix='data/Task10_Colon',augmentation=False,mode='test',choose='mask')
    

    # test_dataset = LiTSVolumeDataset(path_prefix='data/Task03_LITS17',augmentation=False,mode='test',choose='mask')

    # nice_train_loader = DataLoader(train_dataset)
    nice_test_loader = DataLoader(test_dataset)

    # nice_train_loader, nice_test_loader = get_dataloader(args)

    '''checkpoint path and tensorboard'''
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    '''begain training'''
    best_acc = 0.0
    best_tol = 1e4
    best_dice = 0.0
    best_epoch = 0
    

    
    for epoch in range(settings.EPOCH):
        #冻结其他参数，先只训练Unet解码部分
        if epoch == 0:
            for param in net.parameters():
                param.requires_grad = False
        # 解冻Unet和image_encoder
            for p in net.SaliencyMLP.parameters():
                p.requires_grad = True
            for p in net.image_encoder.parameters():
                p.requires_grad = True
        #解冻所有参数
        if epoch == 101:
            for param in net.parameters():
                param.requires_grad = True

        for name,param in net.named_parameters():
            if param.requires_grad == False:
                print("{}没有梯度".format(name))
                logger.info("{}没有梯度".format(name))
                break
    # #正常训练代码
    #     net.train()
    #     time_start = time.time()
    #     loss, prompt_loss, last_temp = train_sam(args, net, encoder_opt,prompt_opt,other_opt,component_opt,nice_train_loader, epoch,encoder_scheduler,prompt_scheduler,other_scheduler,component_scheduler)
    #     logger.info(f'Train loss: {loss}, {prompt_loss}, {last_temp} || @ epoch {epoch}.')
    #     time_end = time.time()
    #     print('time_for_training ', time_end - time_start)

        
        tol , (eiou , edice),hd95 = validation_sam(args,nice_test_loader, epoch, net, writer,method='AFP',yolo_model = None)
        logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice}, HD95:{hd95} || @ epoch {epoch}.')
        if edice > best_dice:
            best_dice = edice
            best_tol = tol
            best_iou = eiou
            best_epoch = epoch
            torch.save(
                {
                    'model': net.state_dict(),
                    'epoch':epoch,
                    'best_dice':best_dice,
                    'bese_iou':best_iou,
                    'encoder_opt':encoder_opt,
                    'prompt_opt':prompt_opt,
                    'other_opt':other_opt
                 }, 
                os.path.join(args.path_helper['ckpt_path'], 'best_epoch.pth'))
        if epoch == settings.EPOCH-1:
            logger.info(f'Latest----------Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
            torch.save(
                {
                    'model': net.state_dict(),
                    'epoch':epoch,
                    'best_dice':edice,
                    'bese_iou':eiou,
                    'encoder_opt':encoder_opt,
                    'prompt_opt':prompt_opt,
                    'other_opt':other_opt
                    }, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))
        logger.info(f'Best-----------------Total score: {best_tol}, IOU: {best_iou}, DICE: {best_dice} || @ epoch {best_epoch}.')

        if epoch % args.val_freq == 0:
            torch.save(
                {
                    'model': net.state_dict(),
                    'epoch':epoch,
                    'best_dice':edice,
                    'bese_iou':eiou,
                    'encoder_opt':encoder_opt,
                    'prompt_opt':prompt_opt,
                    'other_opt':other_opt
                    }, os.path.join(args.path_helper['ckpt_path'], f'epoch{epoch}.pth'))
        
        time.sleep(0.3)

    writer.close()


if __name__ == '__main__':
    main()