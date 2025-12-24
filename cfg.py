import argparse


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam2', help='net type')
    parser.add_argument('-encoder', type=str, default='vit_b', help='encoder type')
    parser.add_argument('-exp_name', default='补充实验', type=str, help='experiment name')
    parser.add_argument('-vis', type=bool, default=False, help='Generate visualisation during validation')
    parser.add_argument('-train_vis', type=bool, default=False, help='Generate visualisation during training')
    parser.add_argument('-prompt', type=str, default='Auto', help='type of prompt, bbox or click or Autobbox or Auto')
    parser.add_argument('-prompt_freq', type=int, default=1, help='frequency of giving prompt in 3D images')
    parser.add_argument('-pretrain', type=str, default=None, help='path of pretrain weights')
    parser.add_argument('-val_freq',type=int,default=5,help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-image_size', type=int, default=512, help='image_size')
    parser.add_argument('-out_size', type=int, default=512, help='output_size')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('-dataset', default='stots' ,type=str,help='dataset name')
    parser.add_argument('-sam_ckpt', type=str, default='checkpoints/service_model/Colon/Ours/0.66.pth', help='sam checkpoint address') #logs/MobileInstanceNorm2d_SAM2_0.669/net.best_epoch.pth #checkpoints/best_epoch.pth   #checkpoints/service_model/SAM-SP-Unet/best_epoch.pth
    parser.add_argument('-sam_config', type=str, default='sam2_hiera_l' , help='sam checkpoint address')
    parser.add_argument('-video_length', type=int, default=8, help='sam checkpoint address')
    parser.add_argument('-b', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('-multimask_output', type=int, default=1 , help='the number of masks output for multi-class segmentation')
    parser.add_argument('-memory_bank_size', type=int, default=16, help='sam 2d memory bank size')
    parser.add_argument('-split_pkl_path', type=str, default='data/stots/new_Training_split.pkl', help='stots_split_pkl')
    parser.add_argument( '-data_path', type=str, default='./data/new_stots',help='The path of segmentation data')
    parser.add_argument( '-yolo_pt', type=str, default='checkpoints/yolov12n.pt',help='The path of yolo model')
    parser.add_argument('-multi_prompt',type=str,default=5,help='提示点的个数')
    parser.add_argument('-depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', default=85, help='image size', required=False)
    parser.add_argument('-n_classes', default=1, help='image size', required=False)
    parser.add_argument('-ModelEmb_path', default=None, help='checkpoints/service_model/MoileUnet_train_IOU_Only_MLP/Model/model_emb.best_epoch.pth', required=False)#checkpoints/service_model/MoileUnet_train_IOU_Only_MLP/Model/model_emb.best_epoch.pth
    parser.add_argument('-Auto_prompter', default='MobileUnet', help='选择自动生成器', required=False) #MobileUnet
    parser.add_argument('-max_epoch', default=100, help='最大迭代轮数', required=False)
    # opt = parser.parse_args() #不调试的时候
    
    opt = parser.parse_args(args=[]) #vscode launch.json调试带参数的程序
    return opt

    
