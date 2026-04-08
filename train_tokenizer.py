import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import models.vqvae as vqvae
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval
import utils.eval_trans as eval_trans
import warnings
warnings.filterwarnings('ignore')

import pdb

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

accelerator = Accelerator()

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

# pdb.set_trace()
##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
#breakpoint()
args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))



# if args.dataname == 'kit' : 
#     dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
#     args.nb_joints = 21
# elif args.dataname == 't2m':
#     dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
#     args.nb_joints = 22
# elif args.dataname == 'motionmillion':
#     dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
#     args.nb_joints = 22
args.nb_joints = 22

#logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')
comp_device = accelerator.device

##### ---- Dataloader ---- #####
train_loader, train_mean, train_std = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        args.motion_type, 
                                        args.text_type,
                                        args.version, 
                                        'train', 
                                        args.debug,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t,
                                        num_workers=args.num_workers,
                                        add_hand=args.add_hand)

val_loader, test_mean, test_std = dataset_TM_eval.MotionMillionFSQDATALoader(args.dataname, True,
                                        args.batch_size,
                                        unit_length=2**args.down_t,
                                        version=args.version,
                                        add_hand=args.add_hand,
                                        motion_type=args.motion_type)
#breakpoint()
##### ---- Network ---- #####
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                   args.nb_code, 
                   args.code_dim, #512
                   args.output_emb_width,
                   args.down_t, #1
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

if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    state_dict = torch.load(args.resume_pth, map_location='cpu')
    ckpt = state_dict["net"]
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    net.load_state_dict(ckpt, strict=True)
net.train()
net.to(comp_device)

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

if args.resume_pth:
    optimizer.load_state_dict(state_dict["optimizer"])
    scheduler.load_state_dict(state_dict["scheduler"])

net, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        net, optimizer, train_loader, val_loader, scheduler
    )

train_loader_iter = cycle(train_loader)

Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints,args.motion_type) #l1_smooth ,22

##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit, avg_activate = 0., 0., 0., 0.

if not args.resume_pth:
    for nb_iter in range(1, args.warm_up_iter):
        
        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
        gt_motion = next(train_loader_iter)
        gt_motion = gt_motion.to(comp_device).float() # (bs, 64, dim)

        if args.quantizer == "FSQ":
            pred_motion, loss_commit, perplexity, activate, _ = net(gt_motion)
        else:
            pred_motion, loss_commit, perplexity, activate, _ = net(gt_motion)
            
        # #预训练夹爪数据不参与计算loss   
        # if args.motion_type == 'vector_272':
        #     #print("预训练阶段,夹爪数据不参与计算loss")
        #     pred_motion=pred_motion[:,:272]
        #     gt_motion=gt_motion[:,:272]
            
        loss_motion = Loss(pred_motion, gt_motion)
        loss_vel = Loss.forward_vel(pred_motion, gt_motion)
        
        
        if args.quantizer in ["LFQ", "BSQ"]:
            loss = loss_motion + loss_commit + args.loss_vel * loss_vel
        elif args.quantizer == "FSQ":
            loss = loss_motion + args.loss_vel * loss_vel
        else:
            loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
        
        if args.use_acc_loss:
            loss_acc = Loss.forward_acc(pred_motion, gt_motion)
            loss = loss + args.acc_loss * loss_acc
        if args.use_acc_vel_loss:
            loss_acc_vel = Loss.forward_acc_vel(pred_motion, gt_motion)
            loss = loss + args.acc_vel_loss * loss_acc_vel
        if args.use_root_loss:
            loss_root = Loss.forward_root(pred_motion, gt_motion)
            loss = loss + args.root_loss * loss_root
            
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()

        avg_recons += loss_motion.item()
        avg_perplexity += perplexity.item()
        avg_commit += loss_commit.item()
        avg_activate += activate.item()

        if nb_iter % args.print_iter ==  0 :
            if accelerator.is_main_process:
                avg_recons /= args.print_iter
                avg_perplexity /= args.print_iter
                avg_commit /= args.print_iter
                avg_activate /= args.print_iter
                
                logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Activate. {avg_activate:.2f}")
            
            avg_recons, avg_perplexity, avg_commit, avg_activate = 0., 0., 0., 0.

##### ---- Training ---- #####
avg_recons, avg_perplexity, avg_commit, avg_activate = 0., 0., 0., 0.

accelerator.wait_for_everyone()
best_mpjpe, writer, logger = eval_trans.evaluation_vqvae_motionmillion(args.out_dir, train_loader, val_loader, net, logger, writer, 0, best_mpjpe=1000, comp_device=comp_device, codebook_size=accelerator.unwrap_model(net).vqvae.quantizer.codebook_size, accelerator=accelerator)


if args.resume_pth:
    start_iter = state_dict["nb_iter"] + 1
else:
    start_iter = 1
print('start training from iter {}'.format(start_iter))
for nb_iter in range(start_iter, args.total_iter + 1):
    
    gt_motion = next(train_loader_iter)
    
    pred_motion, loss_commit, perplexity, activate, _ = net(gt_motion)
    
    # if args.motion_type == 'vector_272':
    #         #print("预训练阶段,夹爪数据不参与计算loss")
    #         pred_motion=pred_motion[:,:272]
    #         gt_motion=gt_motion[:,:272]
            
    loss_motion = Loss(pred_motion, gt_motion)
    loss_vel = Loss.forward_vel(pred_motion, gt_motion)
    
    if args.quantizer == "LFQ":
        loss = loss_motion + loss_commit + args.loss_vel * loss_vel
    else:
        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    
    if args.use_acc_loss:
        loss_acc = Loss.forward_acc(pred_motion, gt_motion)
        loss = loss + args.acc_loss * loss_acc
    if args.use_acc_vel_loss:
        loss_acc_vel = Loss.forward_acc_vel(pred_motion, gt_motion)
        loss = loss + args.acc_vel_loss * loss_acc_vel
    
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    avg_activate += activate.item()

    if nb_iter % args.print_iter ==  0 :
        if accelerator.is_main_process:
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter
            avg_activate /= args.print_iter
            
            writer.add_scalar('./Train/L1', avg_recons, nb_iter)
            writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
            writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
            writer.add_scalar('./Train/Activate', avg_activate, nb_iter)
            
            logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Activate {avg_activate:.2f}")
        
        avg_recons, avg_perplexity, avg_commit, avg_activate = 0., 0., 0., 0.
    
    if nb_iter % args.eval_iter==0 :
        accelerator.wait_for_everyone()
        best_mpjpe, writer, logger = eval_trans.evaluation_vqvae_motionmillion(args.out_dir, train_loader, val_loader, net, logger, writer, nb_iter, best_mpjpe, comp_device=comp_device, codebook_size=accelerator.unwrap_model(net).vqvae.quantizer.codebook_size, accelerator=accelerator)
    
    accelerator.wait_for_everyone()
    if nb_iter % args.save_iter == 0 and accelerator.is_main_process:
        torch.save({'net' : net.state_dict(), 
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                    'nb_iter' : nb_iter}, os.path.join(args.out_dir, f'net_{nb_iter}.pth'))
    if nb_iter % args.save_latest == 0 and accelerator.is_main_process:
        torch.save({'net' : net.state_dict(), 
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                    'nb_iter' : nb_iter}, os.path.join(args.out_dir, f'net_latest.pth'))
    
accelerator.wait_for_everyone()