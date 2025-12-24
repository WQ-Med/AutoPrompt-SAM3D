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
from sam2_train.modeling.backbones.utils import ImagePromptEncoder,MLPHead
from sam2_train.modeling.sam.mask_decoder import MaskDecoder
from sam2_train.modeling.sam.transformer import TwoWayTransformer

import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from func_3d.function_SP import eval_seg,DiceLoss,CombinedLoss,visualize_segmentation
from func_3d.dataset.datasets import ColonVolumeDataset,flatten_collate_fn
from monai.losses import DiceLoss, FocalLoss,DiceCELoss
from torch.utils.data import DataLoader
from sam2_train.utils.misc import generate_prompts_from_saliency_batch,compute_saliency_confidence
import torch.nn.functional as F
from sam2_train.utils.misc import generate_bbox,save_fig

from PIL import Image
import numpy as np 
import json

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

def classification_choose_frame(classification_label,score_threshold,num_threshold,k,d):
    '''
    classification_label:[1,N]
    #根据二分类标签来选择最有概率的帧，如果正分类帧的数量低于num_threshold，直接返回non值
    return flag, prompt_frame_id
    '''
    prompt_id = choose_prompt_id(cls_prob=classification_label,k=k,d_min=d,score_threshold=score_threshold)
    if len(prompt_id) < num_threshold:
        return False,[]
    return True, prompt_id

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    n_val = len(val_loader)  # the number of batch
    all_temp = [0,0]
    all_tot = 0
    mix_res = (0,)*1*2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    windows_size = 12 #数据采集时滑动窗口的长度

    lossfunc = DiceLoss(include_background=False,sigmoid=True,reduction='none')
    save_base_dir = 'data/Task10_Colon/mask_confidence_mask' #存储所有生成的Mask文件 {'name':{'id':mask},'id2':mask2}
    confidence_json = {'positive':{},'negative':{}} #{'positive':{'name':[id,...,]},'negative':{'name':[]}}
    prompt = args.prompt
    with torch.no_grad():
        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for pack in val_loader:
                if len(pack) == 0:
                    continue
                torch.cuda.empty_cache()

                #all的数据集
                '''
                clips = torch.stack([s['clip'] for s in pack]) #(T,C,Z,H,W)
                labels = torch.stack([s["label"] for s in pack]) #(T,Z,H,W)
                frame_has_fg = torch.stack([s["frame_has_fg"] for s in pack])  #(T,Z)
                start_idxs = [s["start_idx"] for s in pack] #一维列表
                name = [s["name"] for s in pack] #一维列表，元素相同
                valid_lengths = [s['valid_length'] for s in pack]
                T_length = clips.shape[0]
                name = name[0]
                start_idx = start_idxs[0]
                os.makedirs(os.path.join(save_base_dir,name), exist_ok=True)
                single_name_confidence_json = {name:{"positive":[],"negative":[]}}
                
                for t in range(T_length):
                    single_confidence_json = {'positive':[],'negative':[]}
                    valid_length = valid_lengths[t]
                    imgs_tensor = clips[t].permute(1,0,2,3)
                    mask_dict = labels[t].to(dtype = torch.float32, device = GPUdevice).detach()
                    start_idx = start_idxs[t]
                    
                    if valid_length < windows_size:
                        imgs_tensor = imgs_tensor[:valid_length]
                        mask_dict = mask_dict[:valid_length]
                    video_length = mask_dict.shape[0]
                '''

                #mask的数据集
                imgs_tensor = pack['clip'].permute(0,2,1,3,4)
                mask_dict = pack['label'].permute(1,0,2,3).to(dtype = torch.float32, device = GPUdevice)
                start_idx = pack['start_idx'].item()
                # print(type(start_idx))
                name = pack['name'][0]
                os.makedirs(os.path.join(save_base_dir,name), exist_ok=True)
                single_name_confidence_json = {name:{"positive":[],"negative":[]}}
                single_confidence_json = {'positive':[],'negative':[]}
                # print(imgs_tensor.shape)
                imgs_tensor = imgs_tensor.to(dtype = torch.float32, device = GPUdevice).squeeze()

                train_state = net.train_init_state(imgs_tensor=imgs_tensor)
                features = net.forward_image(imgs_tensor)
                # ---------------------------------------------用Unet解码器来生成伪mask提示---------------------
                feat32 = features['vision_features']
                feat64 = features['backbone_fpn'][-2]
                feat128 = features['backbone_fpn'][-3]
                classification_label = net.cls_head(feat32)
                # flag , prompt_frame_id = classification_choose_frame(classification_label,score_threshold=0.3,num_threshold=2,k=3,d=1)
                flag = True
                prompt_frame_id = [i for i in range(len(classification_label))]
                # print('置信帧的选择为:',prompt_frame_id)
                if not flag:
                    # mask_dict = mask_dict.unsqueeze(1)
                    # temp = eval_seg(torch.zeros_like(mask_dict).to(device=GPUdevice),mask_dict,threshold)
                    # print(temp[0],temp[1])
                    # sing_image_temp[0] += temp[0]
                    # sing_image_temp[1] += temp[1]
                    continue
                disguise_mask = net.SaliencyMLP(feat128, feat64, feat32)
                
                id_to_prompt_idx = {s:i for i,s in enumerate(prompt_frame_id)}
                confidence_mask = disguise_mask[prompt_frame_id].detach()
                for index in prompt_frame_id:
                    temp = eval_seg(confidence_mask[id_to_prompt_idx[index]].unsqueeze(1),mask_dict[index].unsqueeze(0).unsqueeze(0),threshold=threshold)
                    if 0.6 <temp[1] < 1:
                        single_confidence_json['positive'].append(start_idx+index)
                    else:
                        single_confidence_json['negative'].append(start_idx+index)
                    binary_mask = (confidence_mask[id_to_prompt_idx[index]] > 0.5).squeeze().cpu().numpy().astype(np.uint8) * 255  # 0或255
                    # 保存为PNG（8-bit灰度）
                    Image.fromarray(binary_mask).save(f'{save_base_dir}/{name}/{start_idx+index}.png')
                print('这个窗口的结果是:',single_confidence_json)
                single_name_confidence_json[name]['positive'].extend(single_confidence_json['positive'])
                single_name_confidence_json[name]['negative'].extend(single_confidence_json['negative'])
                if len(single_name_confidence_json[name]['positive']) > 0:
                    confidence_json['positive'][name] = single_name_confidence_json[name]['positive']
                if len(single_name_confidence_json[name]['negative']) > 0:
                    confidence_json['negative'][name] = single_name_confidence_json[name]['negative']
                else:
                    print(f'{name}是没有检测出任何胃癌区域的')
        with open(os.path.join(save_base_dir,'index.json'),'w',encoding='utf-8') as f:
            json.dump(confidence_json, f, ensure_ascii=False, indent=4)

    return all_tot/ n_val , tuple([a/n_val for a in all_temp])


def main():
    torch.multiprocessing.set_start_method('spawn')
    #load_dict
    net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    net.cls_head = MLPHead(input_dim=256*32*32,hidden_dim = 256,output_dim=1)
    checkpoint = torch.load('checkpoints/service_model/二分类_裁剪_0.9/Model/best_epoch.pth')
    model_weights = checkpoint['model']
    net.load_state_dict(model_weights)
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
    test_dataset = ColonVolumeDataset(path_prefix='data/Task10_Colon',augmentation=False,mode='test',choose='mask')

    nice_test_loader = DataLoader(test_dataset)

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
        
        tol , (eiou , edice) = validation_sam(args,nice_test_loader, epoch, net, writer)
        logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
        if epoch == 0:
            break
        
        time.sleep(0.3)

    writer.close()


if __name__ == '__main__':
    main()