from .btcv import BTCV
from .stots import StoTS
from .amos import AMOS
from.Unet_stots import Unet_StoTS
import torchvision.transforms as transforms
from torch.utils.data import DataLoader



def get_dataloader(args):
    # transform_train = transforms.Compose([
    #     transforms.Resize((args.image_size,args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_train_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test = transforms.Compose([
    #     transforms.Resize((args.image_size, args.image_size)),
    #     transforms.ToTensor(),
    # ])

    # transform_test_seg = transforms.Compose([
    #     transforms.Resize((args.out_size,args.out_size)),
    #     transforms.ToTensor(),
    # ])
    
    if args.dataset == 'btcv':
        '''btcv data'''
        btcv_train_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        btcv_test_dataset = BTCV(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(btcv_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(btcv_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'amos':
        '''amos data'''
        amos_train_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt)
        amos_test_dataset = AMOS(args, args.data_path, transform = None, transform_msk= None, mode = 'Test', prompt=args.prompt)

        nice_train_loader = DataLoader(amos_train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
        nice_test_loader = DataLoader(amos_test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        '''end'''
    elif args.dataset == 'stots':
        '''stots data'''
        stots_train_dataset = StoTS(args, args.split_pkl_path,args.data_path, transform = None, transform_msk= None, mode = 'Training', prompt=args.prompt, multi_prompt=args.multi_prompt)
        stots_test_dataset = StoTS(args, args.split_pkl_path,args.data_path, transform = None, transform_msk= None, mode = 'test', prompt=args.prompt, multi_prompt=args.multi_prompt)
        # stots_val_dataset = StoTS(args, './data/stots/Training_split.pkl',args.data_path, transform = None, transform_msk= None, mode = 'val', prompt=args.prompt)
        nice_train_loader = DataLoader(stots_train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
        nice_test_loader = DataLoader(stots_test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    else:
        print("the dataset is not supported now!!!")
        
    return nice_train_loader, nice_test_loader

def Unet_get_dataloader(args):
    if args.dataset == 'stots':
        stots_train_dataset = Unet_StoTS(args,'data/new_stots/Unet/obj_train.txt',args.data_path, transform = None, transform_msk= None, mode = 'Training')
        stots_test_dataset = Unet_StoTS(args, 'data/new_stots/Unet/obj_test.txt', args.data_path,transform = None, transform_msk= None, mode = 'test')
        # stots_val_dataset = StoTS(args, './data/stots/Training_split.pkl',args.data_path, transform = None, transform_msk= None, mode = 'val', prompt=args.prompt)
        nice_train_loader = DataLoader(stots_train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
        nice_test_loader = DataLoader(stots_test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False)

    else:
        print('the dataset is not supported now!!!')
    
    return nice_train_loader , nice_test_loader
