import os
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
import json
import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
from dataset import dataset_tokenize
from tqdm import tqdm
from accelerate import Accelerator
from tqdm import tqdm

 
if __name__ == '__main__':
    
    data_root = '/ssd/caoshiqin/datasets/our_mocap_data/processed_data'
    ##### ---- Exp dirs ---- #####
    args = option_trans.get_args_parser()
    torch.manual_seed(args.seed)
    mean = np.load('/ssd/zhengjiakun/dataset/MotionMillion/MotionMillion/mean_std/vector_272/mean.npy')
    std = np.load('/ssd/zhengjiakun/dataset/MotionMillion/MotionMillion/mean_std/vector_272/std.npy')
    # accelerate
    if args.motion_type == 'vector_274':
        new_dim_mean = np.array([0.0, 0.0], dtype=np.float32)  
        new_dim_std = np.array([1.0, 1.0], dtype=np.float32)    
        mean = np.concatenate([mean, new_dim_mean], axis=0)  # shape (274,)
        std = np.concatenate([std, new_dim_std], axis=0) 
        
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    comp_device = accelerator.device

    ##### ---- Logger ---- #####

    net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate,
                        args.vq_act,
                        args.vq_norm,
                        args.kernel_size,
                        args.use_patcher,
                        args.patch_size,
                        args.patch_method,
                        args.use_attn)

    args.nb_code = net.vqvae.quantizer.codebook_size

    print ('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')['net']
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    net.load_state_dict(ckpt, strict=True)
    net.eval()
    net.to(comp_device)
    
    for folder_name in os.listdir(data_root):
        
        folder_path = os.path.join(data_root, folder_name)
        if args.motion_type == 'vector_272':
            pose_path = os.path.join(folder_path, folder_name+'.npy')
        elif args.motion_type == 'vector_274':
            pose_path = os.path.join(folder_path, 'motion_274.npy')
        else:
            raise ValueError(f"Unsupported motion type: {args.motion_type}")
        
        if not os.path.exists(pose_path):
            print(f"⚠️  文件不存在，跳过：{pose_path}")
            continue
        pose = np.load(pose_path)
        pose = (pose - mean) / std
        pose = torch.from_numpy(pose).float().to(comp_device).unsqueeze(0)
 
        print(f"Processing {pose_path} with shape {pose.shape}")
        #breakpoint()
        with torch.no_grad():
            target = net.encode(pose)
            target = target.cpu().numpy()
            print(f"Encoded shape: {target.shape}")
            output_path = ''
            if args.motion_type == 'vector_274':
                output_path = pjoin(folder_path,'fsq_motion_274.npy')
            elif args.motion_type == 'vector_272':
                output_path = pjoin(folder_path,'fsq_motion_272.npy')
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, target)

   
    
          
