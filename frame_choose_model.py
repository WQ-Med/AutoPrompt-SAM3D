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
from sam2_train.modeling.backbones.utils import ImagePromptEncoder, MLPHead, MaskClassifier
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
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

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

def read_json(path):
    with open(path,'r') as f:
        return json.load(f)

def flatten_data(data_dict, class_label):
    samples = []
    for name, ids in data_dict.items():
        for img_id in ids:
            samples.append((class_label, name, img_id))
    return samples

loss_function = CLSFocalLoss()

def train_sam(args,epoch,net:nn.Module,component_opt,clean_dir=True):
    net.train()
    batch_size = 8
    base_dir = 'data/Task10_Colon/mask_confidence_mask'
    indexjson = read_json('data/Task10_Colon/mask_confidence_mask/index.json')
    positive_samples = flatten_data(indexjson['positive'], 'positive')
    negative_samples = flatten_data(indexjson['negative'], 'negative')
    epoch_loss,all_accuracy,all_precision,all_recall,all_f1_score = 0,0,0,0,0

    random.shuffle(positive_samples)
    random.shuffle(negative_samples)
    negative_samples = negative_samples[:50]

    start,count = 0,0
    while start+batch_size/2 < len(positive_samples):
        print('进度:{}/{}'.format(start,len(positive_samples)))
        p_batch = positive_samples[start:int(start+batch_size/2)]
        n_batch = negative_samples[start:int(start+batch_size/2)]
        
        torch.cuda.empty_cache()
        positive_data = []
        negative_data = []

        for i in range(int(batch_size/2)):
            p_label, p_name, p_id = p_batch[i]
            n_label, n_name, n_id = n_batch[i]

            p_mask = np.array(Image.open(os.path.join(base_dir,p_name,str(p_id)+'.png')))
            n_mask = np.array(Image.open(os.path.join(base_dir,n_name,str(n_id)+'.png')))

            positive_data.append(torch.tensor(p_mask))
            negative_data.append(torch.tensor(n_mask))
        
        data = (
            [(mask, 1) for mask in positive_data] +  # positive 标签为 1
            [(mask, 0) for mask in negative_data]    # negative 标签为 0
            )
        random.shuffle(data)  # 原地打乱
        shuffled_masks, shuffled_labels = zip(*data)  # 解压成两个元组

        images_tensors = torch.stack(shuffled_masks, dim=0).to(dtype = torch.float32, device = GPUdevice).unsqueeze(1)
        label = torch.tensor(shuffled_labels).to(dtype = torch.float32, device = GPUdevice).unsqueeze(1)

        pred = net(images_tensors)
        print('当前指标为:',pred)
        loss = loss_function(pred,label)
        component_opt.zero_grad()
        loss.backward()
        component_opt.step()
        mask_dict_cpu = label.detach().cpu().numpy()
        pred_cpu = (pred>0.5).detach().cpu().numpy()
        # print(mask_dict_cpu,pred_cpu)
        accuracy = accuracy_score(mask_dict_cpu,pred_cpu)
        all_accuracy += accuracy
        precision = precision_score(mask_dict_cpu,pred_cpu)
        all_precision += precision
        recall = recall_score(mask_dict_cpu,pred_cpu)
        all_recall += recall
        f1 = f1_score(mask_dict_cpu,pred_cpu)
        all_f1_score += f1
        print('accuracy:{},P:{},R:{},F1:{}'.format(accuracy,precision,recall,f1))
        start += batch_size
        count += 1
        epoch_loss += loss.item()
    return epoch_loss/count,all_accuracy/count,all_precision/count,all_recall/count,all_f1_score/count


def validation_sam(args,epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    batch_size = 8
    base_dir = 'data/Task10_Colon/mask_confidence_mask'
    indexjson = read_json('data/Task10_Colon/mask_confidence_mask_test/index.json')
    positive_samples = flatten_data(indexjson['positive'], 'positive')
    negative_samples = flatten_data(indexjson['negative'], 'negative')
    epoch_loss,all_accuracy,all_precision,all_recall,all_f1_score = 0,0,0,0,0

    random.shuffle(positive_samples)
    random.shuffle(negative_samples)
    negative_samples = negative_samples[:50]

    start,count = 0,0
    with torch.no_grad():
        while start+batch_size/2 < len(positive_samples):
            print('进度:{}/{}'.format(start,len(positive_samples)))
            p_batch = positive_samples[start:int(start+batch_size/2)]
            n_batch = negative_samples[start:int(start+batch_size/2)]
            
            torch.cuda.empty_cache()
            positive_data = []
            negative_data = []

            for i in range(int(batch_size/2)):
                p_label, p_name, p_id = p_batch[i]
                n_label, n_name, n_id = n_batch[i]

                p_mask = np.array(Image.open(os.path.join(base_dir,p_name,str(p_id)+'.png')))
                n_mask = np.array(Image.open(os.path.join(base_dir,n_name,str(n_id)+'.png')))

                positive_data.append(torch.tensor(p_mask))
                negative_data.append(torch.tensor(n_mask))
            
            data = (
                [(mask, 1) for mask in positive_data] +  # positive 标签为 1
                [(mask, 0) for mask in negative_data]    # negative 标签为 0
                )
            random.shuffle(data)  # 原地打乱
            shuffled_masks, shuffled_labels = zip(*data)  # 解压成两个元组

            images_tensors = torch.stack(shuffled_masks, dim=0).to(dtype = torch.float32, device = GPUdevice).unsqueeze(1)
            label = torch.tensor(shuffled_labels).to(dtype = torch.float32, device = GPUdevice).unsqueeze(1)

            pred = net(images_tensors)
            print('当前指标为:',pred)
            loss = loss_function(pred,label)
            mask_dict_cpu = label.detach().cpu().numpy()
            pred_cpu = (pred>0.5).detach().cpu().numpy()
            # print(mask_dict_cpu,pred_cpu)
            accuracy = accuracy_score(mask_dict_cpu,pred_cpu)
            all_accuracy += accuracy
            precision = precision_score(mask_dict_cpu,pred_cpu)
            all_precision += precision
            recall = recall_score(mask_dict_cpu,pred_cpu)
            all_recall += recall
            f1 = f1_score(mask_dict_cpu,pred_cpu)
            all_f1_score += f1
            print('accuracy:{},P:{},R:{},F1:{}'.format(accuracy,precision,recall,f1))
            start += batch_size
            count += 1
            epoch_loss += loss.item()
        
        return epoch_loss/count,all_accuracy/count,all_precision/count,all_recall/count,all_f1_score/count


def main():
    torch.multiprocessing.set_start_method('spawn')
    #load_dict
    # net = get_network(args, args.net, use_gpu=args.gpu, gpu_device=GPUdevice, distribution = args.distributed)
    # net.cls_head = MLPHead(input_dim=256*32*32,hidden_dim = 256,output_dim=1)
    # checkpoint = torch.load('checkpoints/service_model/二分类_裁剪_0.9/Model/best_epoch.pth')
    # model_weights = checkpoint['model']
    # net.load_state_dict(model_weights)
    net = MaskClassifier()
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
    component_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=1e-4,betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    for epoch in range(settings.EPOCH):
        loss,Accuray,P,R,F1 = train_sam(args,epoch,net,component_opt=component_opt)
        logger.info(f'train loss: {loss}, Accuracy: {Accuray}, F1: {F1} || @ epoch {epoch}.')
        tol , accuracy, P, R, F1 = validation_sam(args,epoch, net, writer)
        logger.info(f'Total loss: {tol}, Accuracy: {accuracy}, P: {P}, R:{R}, F1:{F1} || @ epoch {epoch}.')
        if accuracy > best_acc:
            best_P = P
            best_acc = accuracy
            best_R = R
            best_F1 = F1
            best_epoch = epoch
            torch.save(
                {
                    'model': net.state_dict(),
                    'epoch':epoch,
                    'best_percision':best_P,
                    'component_opt':component_opt
                 }, 
                os.path.join(args.path_helper['ckpt_path'], 'best_epoch'+str(epoch)+'.pth'))
        if epoch == settings.EPOCH-1:
            logger.info(f'Latest----------Total loss: {tol}, Accuracy: {accuracy}, {P},{R},{F1} || @ epoch {epoch}.')
            torch.save(
                {
                    'model': net.state_dict(),
                    'epoch':epoch,
                    'best_percision':best_P,
                    'component_opt':component_opt
                    }, os.path.join(args.path_helper['ckpt_path'], 'latest_epoch.pth'))
        logger.info(f'Best-----------------Total score: loss: {tol}, Accuracy: {best_acc}, {best_P},{best_R},{best_F1} || @ epoch {epoch}.')

        if epoch % args.val_freq == 0:
            torch.save(
                {
                    'model': net.state_dict(),
                    'epoch':epoch,
                    'best_percision':best_P,
                    'component_opt':component_opt
                    }, os.path.join(args.path_helper['ckpt_path'], f'epoch{epoch}.pth'))
        
        time.sleep(0.3)

    writer.close()


if __name__ == '__main__':
    main()