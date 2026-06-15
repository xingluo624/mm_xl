import os
import torch
import numpy as np

from os.path import join as pjoin

import options.option_transformer as option_trans
import models.vqvae as vqvae

from accelerate import Accelerator

 
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
            print(f"Processing {pose_path} with shape {pose.shape}")
            # ========== 准备输出路径 ==========
            output_dir = os.path.join(folder_path, "vqvae_codes")
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(pose_path))[0]
            output_txt = os.path.join(output_dir, f"{base_name}_codes.txt")
            
            # 辅助函数：将 tensor 转为可写入的字符串
            def tensor_to_str(tensor_1d):
                return " ".join(map(str, tensor_1d.squeeze().tolist()))
            
            with torch.no_grad(), open(output_txt, "w") as f:
                f.write(f"# Source: {pose_path}\n")
                f.write(f"# Original shape: {pose.shape}\n")
                f.write(f"# Codebook size: {args.nb_code}\n")
                f.write("="*80 + "\n\n")
                
                # ========== 1. 整段 Pose Encode ==========
                print(f"  🔄 Encoding full sequence...")
                target_full = net.encode(pose)  # [1, T_down, code_dim] 或 [1, num_codes]
                target_full_cpu = target_full.squeeze().cpu().numpy()
                
                f.write("[FULL_SEQUENCE]\n")
                f.write(f"shape: {target_full_cpu.shape}\n")
                f.write(f"data: {tensor_to_str(target_full)}\n\n")
                print(f"  ✅ Full encoded shape: {target_full_cpu.shape}")
                
                # ========== 2. 每 64 帧滑动窗口 Encode ==========
                window_size = 64  # 原始帧数窗口
                stride = 64       # 滑动步长（无重叠）；如需重叠可改为 32 等
                frame_len = pose.shape[1]
                
                f.write(f"[SLIDING_WINDOW]\n")
                f.write(f"window_size: {window_size}, stride: {stride}, total_frames: {frame_len}\n")
                
                window_idx = 0
                for start in range(0, frame_len, stride):
                    end = min(start + window_size, frame_len)
                    window_pose = pose[:, start:end, :]  # [1, window_len, feat_dim]
                    
                    # 跳过过短的片段（可选：如果模型要求最小长度）
                    if window_pose.shape[1] < 8:
                        continue
                        
                    target_win = net.encode(window_pose)
                    target_win_cpu = target_win.squeeze().cpu().numpy()
                    
                    f.write(f"\n[WINDOW_{window_idx:04d}]\n")
                    f.write(f"frame_range: [{start}, {end})\n")
                    f.write(f"shape: {target_win_cpu.shape}\n")
                    f.write(f"data: {tensor_to_str(target_win)}\n")
                    
                    window_idx += 1
                    if window_idx % 10 == 0:
                        print(f"    📦 Processed {window_idx} windows...")
                
                f.write(f"\n# Total windows: {window_idx}\n")
                print(f"  ✅ Saved {window_idx} window codes to {output_txt}")
        

   
    
          
