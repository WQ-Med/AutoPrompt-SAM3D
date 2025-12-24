# train.py
#!/usr/bin/env	python3

import os
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import cfg
# from func_3d import function_SP as function
from func_3d import function_SP_adapter as function
# from func_3d import function
from conf import settings
from func_3d.utils import get_network, set_log_dir, create_logger
from func_3d.dataset import get_dataloader
import torch.nn as nn
from sam2_train.modeling.backbones.utils import ImagePromptEncoder,ClSHead
from sam2_train.modeling.sam.mask_decoder import MaskDecoder
from sam2_train.modeling.sam.transformer import TwoWayTransformer

import multiprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt
from func_3d.function_SP import eval_seg,DiceLoss,CombinedLoss,visualize_segmentation
from func_3d.dataset.datasets import ColonVolumeDataset,flatten_collate_fn
from monai.losses import DiceLoss, FocalLoss
from torch.utils.data import DataLoader
from sam2_train.utils.misc import generate_prompts_from_saliency_batch

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
    hard = 0
    Cls_loss = 0
    Disguise_loss = 0
    epoch_loss = 0
    Final_loss = 0
    has_fig_count = 0 #记录有多少次是有提示帧出现的轮
    ind = 0
    mix_res = (0,)*1*2
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    # train mode
    net.train()
    video_length = args.video_length

    GPUdevice = torch.device('cuda:' + str(args.gpu_device))
    prompt = args.prompt
    prompt_freq = args.prompt_freq

    # lossfunc = criterion_G
    S_lossfunc = DiceLoss(sigmoid=True)
    # S_lossfunc =torch.nn.BCEWithLogitsLoss()
    lossfunc = CombinedLoss(dice_weight=1/21, focal_weight=20 / 21)
    cls_lossfunc = CLSFocalLoss()

    seed = torch.randint(1,11,(1,7))

    with tqdm(total=len(train_loader), desc=f'Epoch {epoch}', unit='img') as pbar:
        for pack in train_loader:
            #定义一个pack的loss
            if len(pack) == 0:
                continue
            pack_cls_loss, pack_disguise_loss, pack_prompt_loss, pack_nonprompt_loss = 0, 0, 0, 0

            torch.cuda.empty_cache()
            clips = torch.stack([s['clip'] for s in pack]).permute(0,2,1,3,4).to(GPUdevice)
            # assert clips.shape[-1] == clips.shape[-2],"读取dataset都错了"
            if clips.shape[-1] != clips.shape[-2]:
                continue
            labels = torch.stack([s["label"] for s in pack]).to(GPUdevice)
            frame_has_fgs = torch.stack([s["frame_has_fg"] for s in pack]).to(GPUdevice)
            start_idxs = [s["start_idx"] for s in pack]
            names = [s["name"] for s in pack]
            for i in range(clips.shape[0]):
                imgs_tensor = clips[i]
                mask_dict = labels[i]
                frame_has_fg = frame_has_fgs[i]
                start_idx = start_idxs[i]
                frame_nums = imgs_tensor.shape[0]

                imgs_tensor = imgs_tensor.squeeze(0)
                imgs_tensor = imgs_tensor.to(dtype = torch.float32, device = GPUdevice)
                
                train_state = net.train_init_state(images=imgs_tensor)
                features = net.forward_image(imgs_tensor)
                # def all_unique_tensors(x):
                #     N = x.shape[0]
                #     for i in range(N):
                #         for j in range(i + 1, N):
                #             if torch.equal(x[i], x[j]):
                #                 return False
                #     return True
                # ---------------------------------------------用二分类来选帧--------------------------
                # vision_features = features['vision_features'] #16*256*32*32
                # video_length = len(vision_features)
                # # print(all_unique_tensors(vision_features))
                # cls_prob = net.cls_head(vision_features).squeeze() #(16,)
                # # cls_prob = torch.sigmoid(cls_prob)
                # prompt_frame_id = choose_prompt_id(cls_prob,k=4,d_min=3)
                # print('提示帧为:',prompt_frame_id)
                # #计算二分类损失
                # cls_loss = cls_lossfunc(cls_prob,frame_has_fg.squeeze())
                # print('二分类损失函数:',cls_loss)
                # print('预测的二分类类别为',cls_prob)
                # print('真实的二分类类别为',frame_has_fg)
                # pack_cls_loss += cls_loss.item()
                # if len(prompt_frame_id) == 0:
                #     encoder_pot.zero_grad()
                #     prompt_opt.zero_grad()
                #     other_opt.zero_grad()
                #     cls_loss.backward()
                #     encoder_pot.step()
                #     prompt_opt.step()
                #     other_opt.step()
                #     component_opt.step()
                #     continue
                # ---------------------------------------------用二分类来选帧--------------------------
                # ---------------------------------------------用Unet解码器来生成伪mask提示---------------------
                # prompt_frame_id = torch.tensor(prompt_frame_id)
                # id_to_prompt_idx = {s:i for i,s in enumerate(prompt_frame_id)}
                # has_fig_count += 1
                # feat32 = vision_features[prompt_frame_id]
                # feat64 = features['backbone_fpn'][-2][prompt_frame_id]
                # feat128 = features['backbone_fpn'][-3][prompt_frame_id]
                # disguise_mask = net.SaliencyMLP(feat128, feat64, feat32)
                # # print(disguise_mask.shape)
                # prompts = generate_prompts_from_saliency_batch(disguise_mask)
                # #计算伪mask损失
                # disguise_loss = lossfunc(disguise_mask,mask_dict[prompt_frame_id].unsqueeze(1)) #(4,1,H,W)
                # pack_disguise_loss += disguise_loss.item()
                # ---------------------------------------------用Unet解码器来生成伪mask提示---------------------

                #---------------------------------------------用Unet解码器来实现二分类以及伪mask提示--------------------------------
                feat32 = features['vision_features']
                video_length = len(feat32)
                feat64 = features['backbone_fpn'][-2]
                feat128 = features['backbone_fpn'][-3]
                disguise_mask, cls_prob = net.SaliencyMLP(feat128, feat64, feat32)
                #计算二分类损失
                cls_loss = cls_lossfunc(cls_prob.squeeze(),frame_has_fg.squeeze())
                print('二分类损失函数:',cls_loss)
                print('预测的二分类类别为',cls_prob.squeeze())
                print('真实的二分类类别为',frame_has_fg)
                prompt_frame_id = choose_prompt_id(cls_prob,k=4,d_min=3)
                print('提示帧为:',prompt_frame_id)
                pack_cls_loss += cls_loss.item()
                if len(prompt_frame_id) == 0:
                    encoder_pot.zero_grad()
                    prompt_opt.zero_grad()
                    other_opt.zero_grad()
                    cls_loss.backward()
                    encoder_pot.step()
                    prompt_opt.step()
                    other_opt.step()
                    component_opt.step()
                    continue
                # prompt_frame_id = torch.tensor(prompt_frame_id)
                id_to_prompt_idx = {s:i for i,s in enumerate(prompt_frame_id)}
                prompts = generate_prompts_from_saliency_batch(disguise_mask[prompt_frame_id])
                disguise_loss = lossfunc(disguise_mask[prompt_frame_id],mask_dict[prompt_frame_id].unsqueeze(1)) #只计算置信帧的损失
                pack_disguise_loss += disguise_loss.item()
                
                 #---------------------------------------------用Unet解码器来实现二分类以及伪mask提示--------------------------------
                # prompt_frame_id = list(range(0, video_length, prompt_freq))
                obj_list = [1.0]

                name = names[i]
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
                                        labels=labels,
                                        clear_old_points=False,
                                    )
                                elif prompt == 'bbox' or prompt == 'Autobbox':
                                    bbox = prompts[id_to_prompt_idx[id]]['box_prompt']
                                    print('当前bbox为:',bbox)
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
                                # elif prompt == 'Image':
                                #     bbox = prompts['box_prompt']
                                #     # if ann_obj_id in bbox_dict[id].keys():
                                #     #     print('当前bbox为:',bbox,bbox_dict[id][ann_obj_id])
                                #     net.train_add_initional_image(
                                #         inference_state=train_state,
                                #         frame_idx = id,
                                #         obj_id = ann_obj_id,
                                #     )
                            except KeyError:
                                _, _, _ = net.train_add_new_mask(
                                    inference_state=train_state,
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    mask = torch.zeros(imgs_tensor.shape[2:]).to(device=GPUdevice),
                                )
                                # if prompt == 'dual':
                                #     return None
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
                    all_loss = cls_loss + 0.5 * disguise_loss + prompt_loss + non_prompt_loss
                    encoder_pot.zero_grad()
                    prompt_opt.zero_grad()
                    other_opt.zero_grad()
                    all_loss.backward()
                    encoder_pot.step()
                    prompt_opt.step()
                    other_opt.step()

                    pbar.set_postfix(**{'loss (batch)': loss})

                total_num = video_length * len(obj_list)
                temp = (pred_iou / total_num, pred_dice / total_num)
                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])
                print('当前{}/{}的clip指标为{},{}'.format(name,start_idx,temp[0],temp[1]))
                net.reset_state(train_state)
            encoder_scheduler.step()
            prompt_scheduler.step()
            other_scheduler.step()
            componet_scheduler.step()
            pbar.update()
    return epoch_loss / len(train_loader),0,0

def validation_sam(args, val_loader, epoch, net: nn.Module, clean_dir=True):
     # eval mode
    net.eval()

    n_val = len(val_loader)  # the number of batch
    mix_res = (0,)*1*2
    tot = 0
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    prompt_freq = args.prompt_freq

    # lossfunc = criterion_G
    lossfunc = DiceLoss(include_background=False,sigmoid=True,reduction='none')
    # lossfunc = paper_loss

    prompt = args.prompt
    with torch.no_grad():
        with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
            for pack in val_loader:
                torch.cuda.empty_cache()
                imgs_tensor = pack['image']
                mask_dict = pack['label']
                name = pack['name']
                print(name)
                if prompt == 'click':
                    pt_dict = pack['pt']
                    point_labels_dict = pack['p_label']
                elif prompt == 'bbox' or prompt == 'Autobbox' or prompt == 'Image':
                    bbox_dict = pack['bbox']
                if len(imgs_tensor.size()) == 5:
                    imgs_tensor = imgs_tensor.squeeze(0)
                frame_id = list(range(imgs_tensor.size(0)))
                
                train_state = net.val_init_state(imgs_tensor=imgs_tensor)
                # prompt_frame_id = list(range(0, len(frame_id), prompt_freq))
                prompt_frame_id = net.train_choose_prompt_id(train_state)
                print('提示帧为:',prompt_frame_id)
                obj_list = []
                for id in frame_id:
                    obj_list += list(mask_dict[id].keys())
                obj_list = list(set(obj_list))
                if len(obj_list) == 0:
                    continue

                name = pack['image_meta_dict']['filename_or_obj']

                for id in prompt_frame_id:
                    for ann_obj_id in obj_list:
                        prompts = net.generate_prompt_auto(train_state,id,mode='eval')[0]
                        try:
                            if prompt == 'click':
                                points = prompts['point_prompt']
                                labels = point_labels_dict[id][ann_obj_id].to(device=GPUdevice)
                                _, _, _ = net.train_add_new_points( #它们都说这是第一帧的处理方式,我可以试一下看看是不是第一帧的处理方式
                                    inference_state=train_state, #我感觉它更像是一种针对于有帧来进行处理的一个东西
                                    frame_idx=id,
                                    obj_id=ann_obj_id,
                                    points=points,
                                    labels=labels,
                                    clear_old_points=False,
                                )
                            elif prompt == 'bbox' or prompt == 'Autobbox' or prompt == 'Auto':
                                bbox = prompts['box_prompt']
                                print('当前bbox为:',bbox,bbox_dict[id][ann_obj_id])
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
                            elif prompt == 'Image':
                                bbox = prompts['box_prompt']
                                if ann_obj_id in bbox_dict[id].keys():
                                    print('当前bbox为:',bbox,bbox_dict[id][ann_obj_id])
                                net.add_initional_image(
                                    inference_state=train_state,
                                    frame_idx = id,
                                    obj_id = ann_obj_id,
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

                loss = 0
                pred_iou = 0
                pred_dice = 0
                need_to_see_flag = False
                for id in frame_id:
                    for ann_obj_id in obj_list:
                        pred = video_segments[id][ann_obj_id]
                        pred = pred.unsqueeze(0)
                        # print(pred,pred.shape)
                        # pred = torch.sigmoid(pred)
                        try:
                            mask = mask_dict[id][ann_obj_id].to(dtype = torch.float32, device = GPUdevice)
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


                total_num = len(frame_id) * len(obj_list)
                loss = loss / total_num
                # print('loss',loss)
                temp = (pred_iou / total_num, pred_dice / total_num)
                tot += loss

                mix_res = tuple([sum(a) for a in zip(mix_res, temp)])

                net.reset_state(train_state)
                pbar.update()

    return tot/ n_val , tuple([a/n_val for a in mix_res])


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
        net.cls_head = ClSHead(hidden_dim=256) #这个待会去计算一下它怎么输出的
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
        + list(net.cls_head.parameters())
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
    component_opt= torch.optim.RMSprop(filter(lambda p: p.requires_grad, new_component_layer),lr=5e-5,eps=1e-08, weight_decay=0)
    
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
    #数据集构造
    train_dataset = ColonVolumeDataset(path_prefix='data/Task10_Colon',augmentation=True,rand_crop_spatial_size=(128,512,512))
    val_dataset = ColonVolumeDataset(path_prefix='data/Task10_Colon',augmentation=False,mode='val',rand_crop_spatial_size=(128,512,512))
    test_dataset = ColonVolumeDataset(path_prefix='data/Task10_Colon',augmentation=False,mode='test',rand_crop_spatial_size=(128,512,512))

    nice_train_loader = DataLoader(train_dataset,collate_fn=flatten_collate_fn)
    nice_test_loader = DataLoader(test_dataset,collate_fn=flatten_collate_fn)

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
    epoch = 1

    
    for epoch in range(settings.EPOCH):
    # 正常训练代码
        net.train()
        time_start = time.time()
        loss, prompt_loss, last_temp = train_sam(args, net, encoder_opt,prompt_opt,other_opt,component_opt,nice_train_loader, epoch,encoder_scheduler,prompt_scheduler,other_scheduler,component_scheduler)
        logger.info(f'Train loss: {loss}, {prompt_loss}, {last_temp} || @ epoch {epoch}.')
        time_end = time.time()
        print('time_for_training ', time_end - time_start)

        '''
        tol , (eiou , edice) = validation_sam(args,nice_test_loader, epoch, net, writer)
        logger.info(f'Total score: {tol}, IOU: {eiou}, DICE: {edice} || @ epoch {epoch}.')
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
        '''
        time.sleep(0.3)

    writer.close()


if __name__ == '__main__':
    main()