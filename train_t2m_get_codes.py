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
    ##### ---- Exp dirs ---- #####
    args = option_trans.get_args_parser()
    torch.manual_seed(args.seed)
    if args.debug:
        args.exp_name = 'debug'
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')

    os.makedirs(args.out_dir, exist_ok = True)

    if args.debug:
        args.print_iter = 1

    # accelerate
    accelerator = Accelerator(mixed_precision=args.mixed_precision, gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    # comp_device = torch.device('cuda')
    comp_device = accelerator.device

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

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

    ##### ---- get code ---- #####
    if args.dataname == 'motionmillion':
        root_dir = "/ssd/zhengjiakun/dataset/MotionMillion/MotionMillion"
        args.vq_dir = os.path.join(root_dir, f'{args.vq_name}')
        args.prob_dir = os.path.join(root_dir, f'{args.vq_name}' + '_prob.npy')
        
    elif args.dataname == 'mocap':
        root_dir = '/ssd/caoshiqin/datasets/our_mocap_data/processed_data'
        args.vq_dir = os.path.join(root_dir, f'{args.vq_name}')
        args.prob_dir = os.path.join(root_dir, f'{args.vq_name}' + '_prob.npy')
    
    # divider --------
    if accelerator.is_main_process:

        if not os.path.exists(args.vq_dir) or not os.path.exists(args.prob_dir):
            logger.info(f"Start to get code from the {args.dataname}!")
            train_loader_token, _, _ = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=2**args.down_t, motion_type=args.motion_type, text_type=args.text_type, version=args.version)
            os.makedirs(args.vq_dir, exist_ok = True)
            
            # initialize the code counts
            code_counts = torch.zeros(args.nb_code + 2, dtype=torch.long)
            total_tokens = 0
            
            # first loop to count the frequency of each code
            for batch in tqdm(train_loader_token):
                pose, name = batch
                bs, seq = pose.shape[0], pose.shape[1]
                pose = pose.to(comp_device).float() # bs, nb_joints, joints_dim, seq_len
                
                with torch.no_grad():
                    target = net.encode(pose)
                    target_with_end = torch.cat([target, torch.ones(target.shape[0], 1).to(target.device) * args.nb_code], dim=1)
                    # count the frequency of each code
                    unique_codes, counts = torch.unique(target_with_end, return_counts=True)
                    for code, count in zip(unique_codes, counts):
                        code_counts[code.long()] += count.item()
                    total_tokens += target_with_end.numel()
                    
                    # save the code results
                    target = target.cpu().numpy()
                    output_path = pjoin(args.vq_dir, name[0] +'.npy')
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    np.save(output_path, target)
            
            # calculate the probability distribution
            code_probs = code_counts.float() / total_tokens
            
            # save the probability distribution
            torch.save(code_probs, args.prob_dir)
            logger.info(f"Code distribution saved to {args.prob_dir}")
        else:
            if accelerator.is_main_process:
                logger.info(f"The code has been saved in {args.vq_dir} before!")
    
    #merge_into_pickle(root_dir, pjoin(root_dir, "split/version1/t2m_60_300/all.txt"))            
