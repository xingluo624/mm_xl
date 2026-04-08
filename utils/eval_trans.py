import os

import clip
import numpy as np
import torch
from scipy import linalg

import itertools
import json
import visualize.plot_3d_global as plot_3d
from visualize.recover_visualize import visualize_smpl_85
from utils.motion_process import recover_from_ric, recover_from_local_position, recover_from_local_rotation
from tqdm import tqdm
import re
import random

def tensorborad_add_video_rot(writer, global_rot, nb_iter, tag, nb_vis=4, title_batch=None, outname=None, fps=60):
    visualize_smpl_85(global_rot, title=title_batch, output_path=outname, fps=fps)

@torch.no_grad()
def compute_perplexity(codebook_size, code_idx) :
    # Calculate new centres
    # code_onehot = torch.zeros(codebook_size, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
    # code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

    # code_count = code_onehot.sum(dim=-1)  # codebook_size
    # prob = code_count / torch.sum(code_count)  
    # perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
    # activate = torch.sum(code_count > 0).float() / codebook_size
    # return perplexity, activate
    return 0,0

@torch.no_grad()
def compute_perplexity_cpu(codebook_size, code_idx) :
    code_idx = code_idx.cpu()
    # Calculate new centres
    code_onehot = torch.zeros(codebook_size, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
    code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

    code_count = code_onehot.sum(dim=-1)  # codebook_size
    prob = code_count / torch.sum(code_count)  
    perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
    activate = torch.sum(code_count > 0).float() / codebook_size
    return perplexity, activate 

def normalize_to_eval_mean_std(data, train_mean, train_std, test_mean, test_std):
    data = data * train_std + train_mean
    data = (data - test_mean) / test_std
    return data



@torch.no_grad()        
def evaluation_vqvae(out_dir, train_loader, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper, comp_device, codebook_size, draw = True, save = True, savegif=False, savenpy=False, accelerator=None) : 
    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []


    motion_annotation_list = []
    motion_pred_list = []


    R_precision_real = torch.tensor([0,0,0], device=comp_device)
    R_precision = torch.tensor([0,0,0], device=comp_device)
    matching_score_real = torch.tensor(0.0, device=comp_device)
    matching_score_pred = torch.tensor(0.0, device=comp_device)

    nb_sample = torch.tensor(0, device=comp_device)

    motion_indices = []

    for batch in tqdm(val_loader):
       
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch

        motion = motion.to(comp_device)

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).to(comp_device)

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            pose_xyz = recover_from_ric(torch.from_numpy(pose).float().to(comp_device), num_joints)

            pose = train_loader.dataset.transform(pose)
            
            pred_pose, loss_commit, perplexity, activate, indices = net(torch.from_numpy(pose).to(comp_device))
            motion_indices.append(indices)

            pred_denorm = torch.from_numpy(train_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())).float().to(comp_device)
            pred_xyz = recover_from_ric(pred_denorm, num_joints)
            
            if savenpy:
                np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
                np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

            pred_pose = val_loader.dataset.transform(train_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy()))
            pred_pose_eval[i:i+1,:m_length[i],:] = torch.from_numpy(pred_pose).to(comp_device)
            
            if accelerator is None or accelerator.is_main_process:
                if i < min(4, bs):
                    draw_org.append(pose_xyz)
                    draw_pred.append(pred_xyz)
                    draw_text.append(caption[i])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    # motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    # motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    # gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    # mu, cov= calculate_activation_statistics(motion_pred_np)

    # diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    # diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    # R_precision_real = R_precision_real / nb_sample
    # R_precision = R_precision / nb_sample

    # matching_score_real = matching_score_real / nb_sample
    # matching_score_pred = matching_score_pred / nb_sample

    if accelerator is not None:
        accelerator.wait_for_everyone()
        motion_annotation_list = accelerator.gather(torch.cat(motion_annotation_list, dim=0))
        motion_pred_list = accelerator.gather(torch.cat(motion_pred_list, dim=0))
        motion_indices = accelerator.gather(torch.cat(motion_indices, dim=0))
        # reduce
        R_precision_real = accelerator.reduce(R_precision_real, reduction="sum")
        R_precision = accelerator.reduce(R_precision, reduction="sum")
        matching_score_real = accelerator.reduce(matching_score_real, reduction="sum")
        matching_score_pred = accelerator.reduce(matching_score_pred, reduction="sum")
        #print(f'nb_sample={nb_sample}')
        nb_sample = accelerator.reduce(nb_sample, reduction="sum")

    if accelerator is None or accelerator.is_main_process:
        #print(f'len(motion_annotation_list){len(motion_annotation_list)}') # 23 ----> 48
        motion_indices = [motion_index.flatten() for motion_index in motion_indices]
        all_motion_indices = torch.cat(motion_indices)
    
        perplexity, activate = compute_perplexity(codebook_size, all_motion_indices.reshape(-1).to(torch.int64))
        
        motion_annotation_np = motion_annotation_list.cpu().numpy()
        #print(f'motion_annotation_np.shape={motion_annotation_np.shape}') # 1472 / 1536 ---> 1530
        #print(f'Last nb_sample={nb_sample}')
        motion_pred_np = motion_pred_list.cpu().numpy()
        gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
        mu, cov= calculate_activation_statistics(motion_pred_np)

        diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
        diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

        R_precision_real = R_precision_real / nb_sample
        R_precision = R_precision / nb_sample

        matching_score_real = matching_score_real / nb_sample
        matching_score_pred = matching_score_pred / nb_sample


        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    
        # activate = all_motion_indices.reshape(-1).to(torch.int64).unique().numel() / codebook_size
        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, PPL. {perplexity}, activate. {activate:.4f}"
        logger.info(msg)
    
    if draw and (accelerator is None or accelerator.is_main_process):
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)
        writer.add_scalar('./Test/PPL', perplexity, nb_iter)
        writer.add_scalar('./Test/activate', activate, nb_iter)

    if accelerator is None or accelerator.is_main_process:
        if fid < best_fid : 
            msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            logger.info(msg)
            best_fid, best_iter = fid, nb_iter
            if save:
                torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

        if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
            msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
            logger.info(msg)
            best_div = diversity
            if save:
                torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

        if R_precision[0] > best_top1 : 
            msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
            logger.info(msg)
            best_top1 = R_precision[0]
            if save:
                torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

        if R_precision[1] > best_top2 : 
            msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
            logger.info(msg)
            best_top2 = R_precision[1]
        
        if R_precision[2] > best_top3 : 
            msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
            logger.info(msg)
            best_top3 = R_precision[2]
        
        if matching_score_pred < best_matching : 
            msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
            logger.info(msg)
            best_matching = matching_score_pred
            if save:
                torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


# use
@torch.no_grad()        
def evaluation_vqvae_motionmillion_1gpu(out_dir, train_loader, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, comp_device, draw = True, save = True, savegif=False, savenpy=False, fps=60, cal_acceleration=False) : 
    net.eval()
    
    draw_org = []
    draw_pred = []
    name_list = []
    
    nb_sample = 0
    mpjpe = 0
    
    if cal_acceleration:
        pred_mean_acceleration_seq = 0
        pred_max_acceleration_seq = 0
        gt_mean_acceleration_seq = 0
        gt_max_acceleration_seq = 0

    motion_indices = []

    for batch in tqdm(val_loader):
        
        if len(batch) == 3:
            motion, m_length, name = batch
        else:
            motion = batch

        motion = motion.to(comp_device)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).to(comp_device)
        
        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            real_pose = pose
            
            pose_xyz = torch.from_numpy(recover_from_local_position(pose.squeeze(), num_joints)).float().to(comp_device).unsqueeze(0)
            
            pose_rot = torch.from_numpy(recover_from_local_rotation(pose.squeeze(), num_joints)).float().to(comp_device).unsqueeze(0)
            
            pose = train_loader.dataset.transform(pose)
            
            pred_pose, loss_commit, perplexity, activate, indices = net(torch.from_numpy(pose).to(comp_device))
            motion_indices.append(indices)
            
            pred_denorm = train_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
            pred_xyz = torch.from_numpy(recover_from_local_position(pred_denorm.squeeze(), num_joints)).float().to(comp_device).unsqueeze(0)
            
            pred_rot = torch.from_numpy(recover_from_local_rotation(pred_denorm.squeeze(), num_joints)).float().to(comp_device).unsqueeze(0)
            
            mpjpe += torch.mean(calculate_mpjpe(pose_xyz[:, :m_length[i]].squeeze(), pred_xyz[:, :m_length[i]].squeeze())).cpu()
            
            
            if cal_acceleration:
                pred_mean_acc, pred_max_acc, gt_mean_acc, gt_max_acc = calculate_acceleration(pose_xyz[:, :m_length[i]].squeeze(), pred_xyz[:, :m_length[i]].squeeze())
                pred_mean_acceleration_seq += torch.mean(pred_mean_acc).cpu()
                pred_max_acceleration_seq += torch.mean(pred_max_acc).cpu()
                gt_mean_acceleration_seq += torch.mean(gt_mean_acc).cpu()
                gt_max_acceleration_seq += torch.mean(gt_max_acc).cpu()
            
            if savenpy:
                np.save(os.path.join(out_dir, name[i].replace("/", "@")+'_gt_85rpr.npy'), pose_rot[0, :m_length[i]].cpu().numpy())
                np.save(os.path.join(out_dir, name[i].replace("/", "@")+'_pred_85rpr.npy'), pred_rot[0, :m_length[i]].cpu().numpy())
            
            pred_pose = val_loader.dataset.transform(train_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy()))
            pred_pose_eval[i:i+1,:m_length[i],:] = torch.from_numpy(pred_pose).to(comp_device)

            if draw:
                draw_org.append(pose_rot)
                draw_pred.append(pred_rot)
                name_list.append(name[i].replace("/", "@"))
        nb_sample += bs

    mpjpe = mpjpe / nb_sample
    if cal_acceleration:
        pred_mean_acceleration_seq = pred_mean_acceleration_seq / nb_sample
        pred_max_acceleration_seq = pred_max_acceleration_seq / nb_sample
        gt_mean_acceleration_seq = gt_mean_acceleration_seq / nb_sample
        gt_max_acceleration_seq = gt_max_acceleration_seq / nb_sample
        print("pred_mean_acceleration_seq: ", pred_mean_acceleration_seq)
        print("pred_max_acceleration_seq: ", pred_max_acceleration_seq)
        print("gt_mean_acceleration_seq: ", gt_mean_acceleration_seq)
        print("gt_max_acceleration_seq: ", gt_max_acceleration_seq)
    print("mpjpe: ", mpjpe)

    motion_indices = [motion_index.flatten() for motion_index in motion_indices]
    
    msg = f"--> \t Eva. Iter {nb_iter} :, MPJPE. {mpjpe}"
    if cal_acceleration:
        msg += f", Pred_mean_acceleration_seq. {pred_mean_acceleration_seq:.3f}, Pred_max_acceleration_seq. {pred_max_acceleration_seq:.3f}, Gt_mean_acceleration_seq. {gt_mean_acceleration_seq:.3f}, Gt_max_acceleration_seq. {gt_max_acceleration_seq:.3f}"
    logger.info(msg)
    
    if draw:
        writer.add_scalar('./Test/MPJPE', mpjpe, nb_iter)
        if cal_acceleration:
            writer.add_scalar('./Test/Pred_mean_acceleration_seq', pred_mean_acceleration_seq, nb_iter)
            writer.add_scalar('./Test/Pred_max_acceleration_seq', pred_max_acceleration_seq, nb_iter)
            writer.add_scalar('./Test/Gt_mean_acceleration_seq', gt_mean_acceleration_seq, nb_iter)
            writer.add_scalar('./Test/Gt_max_acceleration_seq', gt_max_acceleration_seq, nb_iter)
        
        print(len(draw_pred))
        for ii in range(len(draw_pred))[:10]:
            print(ii)
            tensorborad_add_video_rot(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=["test"], outname=[os.path.join(out_dir, 'pred'+name_list[ii]+'.gif')] if savegif else None, fps=fps)
            tensorborad_add_video_rot(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=["test"], outname=[os.path.join(out_dir, 'gt'+name_list[ii]+'.gif')] if savegif else None, fps=fps)

    if mpjpe < best_mpjpe : 
        msg = f"--> --> \t MPJPE Improved from {best_mpjpe:.5f} to {mpjpe:.5f} !!!"
        logger.info(msg)
        best_mpjpe = mpjpe
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_mpjpe.pth'))

    if save:
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    if cal_acceleration:
        return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, writer, logger, pred_mean_acceleration_seq, pred_max_acceleration_seq, gt_mean_acceleration_seq, gt_max_acceleration_seq
    else:
        return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_mpjpe, writer, logger


# use
@torch.no_grad()        
def evaluation_vqvae_motionmillion(out_dir, train_loader, val_loader, net, logger, writer, nb_iter, best_mpjpe, comp_device, codebook_size, draw = True, save = True, savenpy=False, accelerator=None, cal_acceleration=False): 
    net.eval()
    
    mpjpe = torch.tensor(0.0, device=comp_device)
    if cal_acceleration:
        pred_mean_acceleration_seq = torch.tensor(0.0, device=comp_device)
        pred_max_acceleration_seq = torch.tensor(0.0, device=comp_device)
        gt_mean_acceleration_seq = torch.tensor(0.0, device=comp_device)
        gt_max_acceleration_seq = torch.tensor(0.0, device=comp_device)
        
    nb_sample = torch.tensor(0, device=comp_device)

    motion_indices = []
    
    for batch in tqdm(val_loader):
        motion, m_length, name = batch

        motion = motion.to(comp_device)

        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).to(comp_device)

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            pose_xyz = torch.from_numpy(recover_from_local_position(pose.squeeze(), num_joints)).float().to(comp_device).unsqueeze(0)

            pose = train_loader.dataset.transform(pose)
            
            pred_pose, _, perplexity, activate, indices = net(torch.from_numpy(pose).to(comp_device))
            motion_indices.append(indices.squeeze().cpu())

            pred_denorm = train_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
            pred_xyz = torch.from_numpy(recover_from_local_position(pred_denorm.squeeze(), num_joints)).float().to(comp_device).unsqueeze(0)
            mpjpe += torch.mean(calculate_mpjpe(pose_xyz[:].squeeze(), pred_xyz[:].squeeze()))
            
            
            if cal_acceleration:
                pred_mean_acc, pred_max_acc, gt_mean_acc, gt_max_acc = calculate_acceleration(pose_xyz[:].squeeze(), pred_xyz[:].squeeze())
                pred_mean_acceleration_seq += torch.mean(pred_mean_acc)
                pred_max_acceleration_seq += torch.max(pred_max_acc)
                gt_mean_acceleration_seq += torch.mean(gt_mean_acc)
                gt_max_acceleration_seq += torch.max(gt_max_acc)
            
            if savenpy:
                np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
                np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

            pred_pose = val_loader.dataset.transform(train_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy()))
            pred_pose_eval[i:i+1,:m_length[i],:] = torch.from_numpy(pred_pose).to(comp_device)
            
        nb_sample = nb_sample + bs
    
    if accelerator is None or accelerator.is_main_process:
        motion_indices = [motion_index.flatten() for motion_index in motion_indices]
        all_motion_indices = torch.cat(motion_indices)
    
        perplexity, activate = compute_perplexity(codebook_size, all_motion_indices.reshape(-1).to(torch.int64))
        mpjpe = mpjpe / nb_sample
        if cal_acceleration:
            pred_mean_acceleration_seq = pred_mean_acceleration_seq / nb_sample
            pred_max_acceleration_seq = pred_max_acceleration_seq / nb_sample
            gt_mean_acceleration_seq = gt_mean_acceleration_seq / nb_sample
            gt_max_acceleration_seq = gt_max_acceleration_seq / nb_sample
        msg = f"--> \t Eva. Iter {nb_iter} :, MPJPE. {mpjpe:.4f} PPL. {perplexity} Activate. {activate:.4f}"
        if cal_acceleration:
            msg += f" Pred Mean Accel. {pred_mean_acceleration_seq:.4f} Pred Max Accel. {pred_max_acceleration_seq:.4f} GT Mean Accel. {gt_mean_acceleration_seq:.4f} GT Max Accel. {gt_max_acceleration_seq:.4f}"
        logger.info(msg)
    
    
    if draw and (accelerator is None or accelerator.is_main_process):
        writer.add_scalar('./Test/PPL', perplexity, nb_iter)
        writer.add_scalar('./Test/activate', activate, nb_iter)
        writer.add_scalar('./Test/MPJPE', mpjpe, nb_iter)
        if cal_acceleration:
            writer.add_scalar('./Test/Pred Mean Accel', pred_mean_acceleration_seq, nb_iter)
            writer.add_scalar('./Test/Pred Max Accel', pred_max_acceleration_seq, nb_iter)
            writer.add_scalar('./Test/GT Mean Accel', gt_mean_acceleration_seq, nb_iter)
            writer.add_scalar('./Test/GT Max Accel', gt_max_acceleration_seq, nb_iter)
    if accelerator is None or accelerator.is_main_process:
        if mpjpe < best_mpjpe : 
            msg = f"--> --> \t MPJPE Improved from {best_mpjpe:.5f} to {mpjpe:.5f} !!!"
            logger.info(msg)
            best_mpjpe = mpjpe
            if save:
                torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_mpjpe.pth'))

        # if save:
        #     torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_mpjpe, writer, logger


@torch.no_grad()        
def evaluation_transformer(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, comp_device, text_encode, text_sum_way, draw = True, save = True, savegif=False, accelerator=None) : 

    trans.eval()
    # nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = torch.tensor([0,0,0], device=comp_device)
    R_precision = torch.tensor([0,0,0], device=comp_device)
    matching_score_real = torch.tensor(0.0, device=comp_device)
    matching_score_pred = torch.tensor(0.0, device=comp_device)
    # R_precision_real = 0
    # R_precision = 0
    # matching_score_real = 0
    # matching_score_pred = 0


    nb_sample = torch.tensor(0, device=comp_device)

    for i in range(1):
        for batch in tqdm(val_loader):
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22

            if text_encode == 'clip':
                text = clip.tokenize(clip_text, truncate=True).to(comp_device)
                feat_clip_text = clip_model.encode_text(text).float() # (bs, 512)
                feat_clip_text = feat_clip_text.unsqueeze(1)
                y_mask = torch.ones((feat_clip_text.shape[0], feat_clip_text.shape[1])).to(comp_device)
               
                assert text_sum_way is None
            elif text_encode == 'flan-t5-xxl':
                tokenizer, text_encoder = clip_model
                cap_inputs = tokenizer(clip_text, padding=True, truncation=True, return_tensors="pt")
                y_mask = cap_inputs.attention_mask.to(device=comp_device)
                feat_clip_text = text_encoder(
                    input_ids=cap_inputs.input_ids.to(comp_device), 
                    attention_mask=cap_inputs.attention_mask.to(comp_device), output_hidden_states=False
                ).last_hidden_state
            elif text_encode == 'flan-t5-xl':
                tokenizer, text_encoder = clip_model
                cap_inputs = tokenizer(clip_text, padding=True, truncation=True, return_tensors="pt")
                y_mask = cap_inputs.attention_mask.to(device=comp_device)
                feat_clip_text = text_encoder(
                    input_ids=cap_inputs.input_ids.to(comp_device), 
                    attention_mask=cap_inputs.attention_mask.to(comp_device), output_hidden_states=False
                ).last_hidden_state #(bs, word_nb, 2048)
            else: 
                raise NotImplementedError

            if text_sum_way == 'cls':
                feat_clip_text = feat_clip_text[:, 0, :]
            elif text_sum_way == 'mean':
                feat_clip_text = (feat_clip_text * y_mask.unsqueeze(-1)).sum(dim=1) / y_mask.sum(dim=1, keepdim=True)
            elif text_sum_way == 'sum':
                feat_clip_text = (feat_clip_text * y_mask.unsqueeze(-1)).sum(dim=1)

            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(comp_device)
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                # try:
                if accelerator is not None:
                    index_motion = accelerator.unwrap_model(trans).sample(feat_clip_text[k:k+1], y_mask[k:k+1], False)
                    pred_pose = accelerator.unwrap_model(net).forward_decoder(index_motion)

                else:
                    index_motion = trans.sample(feat_clip_text[k:k+1],y_mask[k:k+1], False)
                    pred_pose = net.forward_decoder(index_motion)
                # except:
                #     index_motion = torch.ones(1,1).to(comp_device).long()

                
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
                # denorm to original domain and norm to eval domain
               


                if draw and (accelerator is None or accelerator.is_main_process):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    # pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().to(comp_device), num_joints)
                    pred_xyz = torch.from_numpy(recover_from_local_position(pred_denorm.squeeze(), num_joints)).float().to(comp_device).unsqueeze(0)

                    if i == 0 and k < 4:
                        draw_pred.append(pred_xyz)
                        draw_text_pred.append(clip_text[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            # if accelerator is not None:
            #     # all_et_pred = accelerator.gather(et_pred)
            #     # all_em_pred = accelerator.gather(em_pred)
            #     all_et_pred, all_em_pred = accelerator.gather_for_metrics((et_pred, em_pred))
            # else:
            #     all_et_pred = et_pred
            #     all_em_pred = em_pred
            
            if i == 0:
                pose = pose.to(comp_device).float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)

                # if accelerator is not None:
                #     # all_em = accelerator.gather(em)
                #     # all_et = accelerator.gather(et)
                #     all_et, all_em = accelerator.gather_for_metrics((et, em))
                #     # print(bs)
                #     # print(em.shape)
                #     # print(all_em.shape)
                # else:
                #     all_em = em
                #     all_et = et


                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw and (accelerator is None or accelerator.is_main_process):
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    # pose_xyz = recover_from_ric(torch.from_numpy(pose).float().to(comp_device), num_joints)
                    pose_xyz = torch.from_numpy(recover_from_local_position(pose.squeeze(), num_joints)).float().to(comp_device).unsqueeze(0)

                    for j in range(min(4, bs)):
                        draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                        draw_text.append(clip_text[j])

                if accelerator is None or accelerator.is_main_process:
                    temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                    R_precision_real += torch.tensor(temp_R, device=comp_device)
                    matching_score_real += torch.tensor(temp_match, device=comp_device)
                    temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                    R_precision += torch.tensor(temp_R, device=comp_device)
                    matching_score_pred += torch.tensor(temp_match, device=comp_device)

                    nb_sample += et.shape[0]
                    
    if accelerator is not None:
        accelerator.wait_for_everyone()

        motion_annotation_list = accelerator.gather(torch.cat(motion_annotation_list, dim=0))
        motion_pred_list = accelerator.gather(torch.cat(motion_pred_list, dim=0))
        # reduce
        R_precision_real = accelerator.reduce(R_precision_real, reduction="sum")
        R_precision = accelerator.reduce(R_precision, reduction="sum")
        matching_score_real = accelerator.reduce(matching_score_real, reduction="sum")
        matching_score_pred = accelerator.reduce(matching_score_pred, reduction="sum")
        print(nb_sample)
        nb_sample = accelerator.reduce(nb_sample, reduction="sum")
        print(nb_sample)
    else:
        motion_annotation_list = torch.cat(motion_annotation_list, dim=0)
        motion_pred_list = torch.cat(motion_pred_list, dim=0)

    if accelerator is None or accelerator.is_main_process:
        print(len(motion_annotation_list)) # 23 ----> 48
        motion_annotation_np = motion_annotation_list.cpu().numpy()
        print(motion_annotation_np.shape) # 1472 / 1536 ---> 1530
        print(nb_sample)
        motion_pred_np = motion_pred_list.cpu().numpy()
        gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
        mu, cov= calculate_activation_statistics(motion_pred_np)

        diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
        diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

        R_precision_real = R_precision_real / nb_sample
        R_precision = R_precision / nb_sample

        matching_score_real = matching_score_real / nb_sample
        matching_score_pred = matching_score_pred / nb_sample


        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
        logger.info(msg)
    
    
    if draw and (accelerator is None or accelerator.is_main_process):
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    if accelerator is None or accelerator.is_main_process:
        if fid < best_fid : 
            msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            logger.info(msg)
            best_fid, best_iter = fid, nb_iter
            if save:
                torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
        
        if matching_score_pred < best_matching : 
            msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
            logger.info(msg)
            best_matching = matching_score_pred

        if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
            msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
            logger.info(msg)
            best_div = diversity

        if R_precision[0] > best_top1 : 
            msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
            logger.info(msg)
            best_top1 = R_precision[0]

        if R_precision[1] > best_top2 : 
            msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
            logger.info(msg)
            best_top2 = R_precision[1]
        
        if R_precision[2] > best_top3 : 
            msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
            logger.info(msg)
            best_top3 = R_precision[2]

        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


# use
@torch.no_grad()        
def evaluation_transformer_motionmillion(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, comp_device, text_encode, text_sum_way, draw = True, save = True, savegif=False, accelerator=None) : 

    trans.eval()
    
    motion_annotation_list = []  #真实运动序列
    motion_pred_list = [] #生成运动序列
    R_precision_real = torch.tensor([0,0,0], device=comp_device)
    R_precision = torch.tensor([0,0,0], device=comp_device)
    matching_score_real = torch.tensor(0.0, device=comp_device)
    matching_score_pred = torch.tensor(0.0, device=comp_device)


    nb_sample = torch.tensor(0, device=comp_device) #统计验证集样本总数
    
    for i in range(1):
        global_cnt = 0
        
        for batch in tqdm(val_loader):
            global_cnt += 1
            _, _, clip_text, _, pose, m_length, _, name = batch
            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22

            if text_encode == 'clip':
                text = clip.tokenize(clip_text, truncate=True).to(comp_device)
                feat_clip_text = clip_model.encode_text(text).float() # (bs, 512)
                feat_clip_text = feat_clip_text.unsqueeze(1)
                y_mask = torch.ones((feat_clip_text.shape[0], feat_clip_text.shape[1])).to(comp_device)
                assert text_sum_way is None
            elif text_encode == 'flan-t5-xxl':
                tokenizer, text_encoder = clip_model
                cap_inputs = tokenizer(clip_text, padding=True, truncation=True, return_tensors="pt")
                y_mask = cap_inputs.attention_mask.to(device=comp_device)
                feat_clip_text = text_encoder(
                    input_ids=cap_inputs.input_ids.to(comp_device), 
                    attention_mask=cap_inputs.attention_mask.to(comp_device), output_hidden_states=False
                ).last_hidden_state
            elif text_encode == 'flan-t5-xl':
                tokenizer, text_encoder = clip_model
                cap_inputs = tokenizer(clip_text, padding=True, truncation=True, return_tensors="pt")
                y_mask = cap_inputs.attention_mask.to(device=comp_device)
                feat_clip_text = text_encoder(
                    input_ids=cap_inputs.input_ids.to(comp_device), 
                    attention_mask=cap_inputs.attention_mask.to(comp_device), output_hidden_states=False
                ).last_hidden_state #(bs, word_nb, 2048)
            else: 
                raise NotImplementedError

            if text_sum_way == 'cls':
                feat_clip_text = feat_clip_text[:, 0, :]
            elif text_sum_way == 'mean':
                feat_clip_text = (feat_clip_text * y_mask.unsqueeze(-1)).sum(dim=1) / y_mask.sum(dim=1, keepdim=True)
            elif text_sum_way == 'sum':
                feat_clip_text = (feat_clip_text * y_mask.unsqueeze(-1)).sum(dim=1)

            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(comp_device)
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                
                if accelerator is not None:
                    index_motion = accelerator.unwrap_model(trans).sample(feat_clip_text[k:k+1], y_mask[k:k+1], False)
                    pred_pose = accelerator.unwrap_model(net).forward_decoder(index_motion)

                else:
                    index_motion = trans.sample(feat_clip_text[k:k+1],y_mask[k:k+1], False)
                    pred_pose = net.forward_decoder(index_motion)
                
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
            #text\motion的embedding
            et_pred, em_pred = eval_wrapper.get_co_embeddings(clip_text, torch.from_numpy(val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())).to(comp_device), pred_len)

            
            if i == 0:
                pose = pose.to(comp_device).float()
                
                et, em = eval_wrapper.get_co_embeddings(clip_text, torch.from_numpy(val_loader.dataset.inv_transform(pose.detach().cpu().numpy())).to(comp_device), m_length)

                motion_annotation_list.append(em.cpu())
                motion_pred_list.append(em_pred.cpu())

                if accelerator is None or accelerator.is_main_process:
                    temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                    R_precision_real += torch.tensor(temp_R, device=comp_device)
                    matching_score_real += torch.tensor(temp_match, device=comp_device)
                    temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                    R_precision += torch.tensor(temp_R, device=comp_device)
                    matching_score_pred += torch.tensor(temp_match, device=comp_device)

                    nb_sample += et.shape[0]
                    
    if accelerator is not None:
        accelerator.wait_for_everyone()
        #聚合多卡的运动嵌入
        motion_annotation_list = accelerator.gather(torch.cat(motion_annotation_list, dim=0))
        motion_pred_list = accelerator.gather(torch.cat(motion_pred_list, dim=0))
        # reduce 多卡指标求和
        R_precision_real = accelerator.reduce(R_precision_real, reduction="sum")
        R_precision = accelerator.reduce(R_precision, reduction="sum")
        matching_score_real = accelerator.reduce(matching_score_real, reduction="sum")
        matching_score_pred = accelerator.reduce(matching_score_pred, reduction="sum")
        print(nb_sample)
        nb_sample = accelerator.reduce(nb_sample, reduction="sum")
        print(nb_sample)
    else:
        motion_annotation_list = torch.cat(motion_annotation_list, dim=0)
        motion_pred_list = torch.cat(motion_pred_list, dim=0)

    if accelerator is None or accelerator.is_main_process:
        print(len(motion_annotation_list)) # 23 ----> 48
        motion_annotation_np = motion_annotation_list.cpu().numpy()
        print(motion_annotation_np.shape) # 1472 / 1536 ---> 1530
        print(nb_sample)
        motion_pred_np = motion_pred_list.cpu().numpy()
        # 计算运动嵌入的均值（gt_mu）和协方差（gt_cov）
        gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
        mu, cov= calculate_activation_statistics(motion_pred_np)

        diversity_real = 0.0
        diversity = 0.0

        R_precision_real = R_precision_real / nb_sample
        R_precision = R_precision / nb_sample

        matching_score_real = matching_score_real / nb_sample
        matching_score_pred = matching_score_pred / nb_sample
        #计算FID（弗雷歇距离，衡量真实与生成分布相似度）
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
        logger.info(msg)
    

    if accelerator is None or accelerator.is_main_process:
        if fid < best_fid : 
            msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            logger.info(msg)
            best_fid, best_iter = fid, nb_iter
            if save:
                torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
        
        if matching_score_pred < best_matching : 
            msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
            logger.info(msg)
            best_matching = matching_score_pred

        if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
            msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
            logger.info(msg)
            best_div = diversity

        if R_precision[0] > best_top1 : 
            msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
            logger.info(msg)
            best_top1 = R_precision[0]

        if R_precision[1] > best_top2 : 
            msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
            logger.info(msg)
            best_top2 = R_precision[1]
        
        if R_precision[2] > best_top3 : 
            msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
            logger.info(msg)
            best_top3 = R_precision[2]

        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger

@torch.no_grad()        
def evaluation_qwen3_motionmillion(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, tokenizer, eval_wrapper, comp_device, draw = True, save = True, savegif=False, accelerator=None,promptype='tf2m') : 

    trans.eval()
    
    motion_annotation_list = []  #真实运动序列
    motion_pred_list = [] #生成运动序列
    R_precision_real = torch.tensor([0,0,0], device=comp_device)
    R_precision = torch.tensor([0,0,0], device=comp_device)
    matching_score_real = torch.tensor(0.0, device=comp_device)
    matching_score_pred = torch.tensor(0.0, device=comp_device)


    nb_sample = torch.tensor(0, device=comp_device) #统计验证集样本总数
    
    for i in range(1):
        global_cnt = 0

        small_val_loader = itertools.islice(val_loader, 10)


        for batch in tqdm(small_val_loader):
            global_cnt += 1
            _, _, clip_text, _, pose, m_length, _, name = batch
            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22
            
            instructions = "template_pretrain.json"
            instructions = json.load(open(instructions, 'r'))
            tasks = []
            for task in instructions.keys():
                for subtask in instructions[task].keys():
                    tasks.append(instructions[task][subtask])

            
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(comp_device)
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                if promptype == 'tf2m':
                    frame = str(m_length[k])
                    task = tasks[3]
                    prompt = task["input"][0].replace("<Caption_Placeholder>", clip_text[k]).replace("<Frame_Placeholder>",frame)
                else:
                    pass
                
                messages = [
                    {"role": "user", "content": prompt}
                ]
            
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(trans.device)

                with torch.no_grad():
                    generated_ids  = trans.generate(
                        **model_inputs,
                        max_new_tokens=512,           
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
                content = tokenizer.decode(output_ids, skip_special_tokens=True)
                #breakpoint()
                motion_pattern = r"<motion_id_(\d+)>"
                all_matches = re.findall(motion_pattern, content)

                all_indices = [int(x) for x in all_matches]
                try:
                    start_token = int(net.vqvae.quantizer.codebook_size)
                    end_token = start_token + 1
                    start_idx = all_indices.index(start_token)
                    end_idx = all_indices.index(end_token, start_idx + 1)  # 从 start 后找 end
                    index_motion = all_indices[start_idx + 1 : end_idx]
                    index_motion = torch.tensor(index_motion, device=comp_device).unsqueeze(0)
                    pred_pose = net.forward_decoder(index_motion)

                    cur_len = pred_pose.shape[1]

                    pred_len[k] = min(cur_len, seq)
                    pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
                except:
                    continue
            #text\motion的embedding
            et_pred, em_pred = eval_wrapper.get_co_embeddings(clip_text, torch.from_numpy(val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())).to(comp_device), pred_len)

            
            if i == 0:
                pose = pose.to(comp_device).float()
                
                et, em = eval_wrapper.get_co_embeddings(clip_text, torch.from_numpy(val_loader.dataset.inv_transform(pose.detach().cpu().numpy())).to(comp_device), m_length)

                motion_annotation_list.append(em.cpu())
                motion_pred_list.append(em_pred.cpu())

                if accelerator is None or accelerator.is_main_process:
                    temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                    R_precision_real += torch.tensor(temp_R, device=comp_device)
                    matching_score_real += torch.tensor(temp_match, device=comp_device)
                    temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                    R_precision += torch.tensor(temp_R, device=comp_device)
                    matching_score_pred += torch.tensor(temp_match, device=comp_device)

                    nb_sample += et.shape[0]
                    
    if accelerator is not None:
        accelerator.wait_for_everyone()
        #聚合多卡的运动嵌入
        motion_annotation_list = accelerator.gather(torch.cat(motion_annotation_list, dim=0))
        motion_pred_list = accelerator.gather(torch.cat(motion_pred_list, dim=0))
        # reduce 多卡指标求和
        R_precision_real = accelerator.reduce(R_precision_real, reduction="sum")
        R_precision = accelerator.reduce(R_precision, reduction="sum")
        matching_score_real = accelerator.reduce(matching_score_real, reduction="sum")
        matching_score_pred = accelerator.reduce(matching_score_pred, reduction="sum")
        print(nb_sample)
        nb_sample = accelerator.reduce(nb_sample, reduction="sum")
        print(nb_sample)
    else:
        motion_annotation_list = torch.cat(motion_annotation_list, dim=0)
        motion_pred_list = torch.cat(motion_pred_list, dim=0)

    if accelerator is None or accelerator.is_main_process:
        print(len(motion_annotation_list)) # 23 ----> 48
        motion_annotation_np = motion_annotation_list.cpu().numpy()
        print(motion_annotation_np.shape) # 1472 / 1536 ---> 1530
        print(nb_sample)
        motion_pred_np = motion_pred_list.cpu().numpy()
        # 计算运动嵌入的均值（gt_mu）和协方差（gt_cov）
        gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
        mu, cov= calculate_activation_statistics(motion_pred_np)

        diversity_real = 0.0
        diversity = 0.0

        R_precision_real = R_precision_real / nb_sample
        R_precision = R_precision / nb_sample

        matching_score_real = matching_score_real / nb_sample
        matching_score_pred = matching_score_pred / nb_sample
        #计算FID（弗雷歇距离，衡量真实与生成分布相似度）
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
        logger.info(msg)
    

    if accelerator is None or accelerator.is_main_process:
        if fid < best_fid : 
            msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            logger.info(msg)
            best_fid, best_iter = fid, nb_iter
            if save:
                torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
        
        if matching_score_pred < best_matching : 
            msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
            logger.info(msg)
            best_matching = matching_score_pred

        if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
            msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
            logger.info(msg)
            best_div = diversity

        if R_precision[0] > best_top1 : 
            msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
            logger.info(msg)
            best_top1 = R_precision[0]

        if R_precision[1] > best_top2 : 
            msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
            logger.info(msg)
            best_top2 = R_precision[1]
        
        if R_precision[2] > best_top3 : 
            msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
            logger.info(msg)
            best_top3 = R_precision[2]

        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger



@torch.no_grad()        
def evaluation_qwen3vl_motionmillion(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, processor, eval_wrapper, comp_device, draw = True, save = True, savegif=False, accelerator=None) : 

    trans.eval()
    
    motion_annotation_list = []  #真实运动序列
    motion_pred_list = [] #生成运动序列
    R_precision_real = torch.tensor([0,0,0], device=comp_device)
    R_precision = torch.tensor([0,0,0], device=comp_device)
    matching_score_real = torch.tensor(0.0, device=comp_device)
    matching_score_pred = torch.tensor(0.0, device=comp_device)


    nb_sample = torch.tensor(0, device=comp_device) #统计验证集样本总数
    
    for i in range(1):
        global_cnt = 0

        #small_val_loader = itertools.islice(val_loader, 400)


        for batch in tqdm(val_loader):
            global_cnt += 1
            _, _, clip_text, _, pose, m_length, _, name = batch
            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22

            #breakpoint()
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(comp_device)
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                typechoice=['tf2m','t2m']
                #typechoice=['tf2m']
                promptype = random.choice(typechoice)
                if promptype == 'tf2m':
                    frame = str(m_length[k])
                    input_template = "Give me a motion that lasts for approximately <Frame_Placeholder> frames. The caption is: <Caption_Placeholder>"
                    prompt = input_template.replace("<Caption_Placeholder>", clip_text[k]).replace("<Frame_Placeholder>",frame)
                elif promptype == 't2m':
                    input_template = "Generate a motion that reflects the text: <Caption_Placeholder>"
                    prompt = input_template.replace("<Caption_Placeholder>", clip_text[k])
                else:
                    pass  
                
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]  
                    }
                ]
            
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = processor(
                    text=[text],
                    return_tensors="pt"
                ).to(trans.device)

                with torch.no_grad():
                    generated_ids = trans.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=processor.tokenizer.eos_token_id,  # 改用processor的tokenizer
                    eos_token_id=processor.tokenizer.eos_token_id
                )
                
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                content = processor.decode(output_ids, skip_special_tokens=True)
                #breakpoint()
                motion_pattern = r"<motion_id_(\d+)>"
                all_matches = re.findall(motion_pattern, content)

                all_indices = [int(x) for x in all_matches]
                try:
                    start_token = int(net.vqvae.quantizer.codebook_size)
                    end_token = start_token + 1
                    start_idx = all_indices.index(start_token)
                    end_idx = all_indices.index(end_token, start_idx + 1)  # 从 start 后找 end
                    index_motion = all_indices[start_idx + 1 : end_idx]
                    index_motion = torch.tensor(index_motion, device=comp_device).unsqueeze(0)
                    pred_pose = net.forward_decoder(index_motion)

                    cur_len = pred_pose.shape[1]

                    pred_len[k] = min(cur_len, seq)
                    pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
                except:
                    continue
            #text\motion的embedding
            #breakpoint()
            et_pred, em_pred = eval_wrapper.get_co_embeddings(clip_text, torch.from_numpy(val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())).to(comp_device), pred_len)

            
            if i == 0:
                pose = pose.to(comp_device).float()
                
                et, em = eval_wrapper.get_co_embeddings(clip_text, torch.from_numpy(val_loader.dataset.inv_transform(pose.detach().cpu().numpy())).to(comp_device), m_length)

                motion_annotation_list.append(em.cpu())
                motion_pred_list.append(em_pred.cpu())

                if accelerator is None or accelerator.is_main_process:
                    temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                    R_precision_real += torch.tensor(temp_R, device=comp_device)
                    matching_score_real += torch.tensor(temp_match, device=comp_device)
                    temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                    R_precision += torch.tensor(temp_R, device=comp_device)
                    matching_score_pred += torch.tensor(temp_match, device=comp_device)

                    nb_sample += et.shape[0]
                    
    if accelerator is not None:
        accelerator.wait_for_everyone()
        #聚合多卡的运动嵌入
        motion_annotation_list = accelerator.gather(torch.cat(motion_annotation_list, dim=0))
        motion_pred_list = accelerator.gather(torch.cat(motion_pred_list, dim=0))
        # reduce 多卡指标求和
        R_precision_real = accelerator.reduce(R_precision_real, reduction="sum")
        R_precision = accelerator.reduce(R_precision, reduction="sum")
        matching_score_real = accelerator.reduce(matching_score_real, reduction="sum")
        matching_score_pred = accelerator.reduce(matching_score_pred, reduction="sum")
        print(nb_sample)
        nb_sample = accelerator.reduce(nb_sample, reduction="sum")
        print(nb_sample)
    else:
        motion_annotation_list = torch.cat(motion_annotation_list, dim=0)
        motion_pred_list = torch.cat(motion_pred_list, dim=0)

    if accelerator is None or accelerator.is_main_process:
        print(len(motion_annotation_list)) # 23 ----> 48
        motion_annotation_np = motion_annotation_list.cpu().numpy()
        print(motion_annotation_np.shape) # 1472 / 1536 ---> 1530
        print(nb_sample)
        motion_pred_np = motion_pred_list.cpu().numpy()
        # 计算运动嵌入的均值（gt_mu）和协方差（gt_cov）
        gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
        mu, cov= calculate_activation_statistics(motion_pred_np)

        diversity_real = 0.0
        diversity = 0.0

        R_precision_real = R_precision_real / nb_sample
        R_precision = R_precision / nb_sample

        matching_score_real = matching_score_real / nb_sample
        matching_score_pred = matching_score_pred / nb_sample
        #计算FID（弗雷歇距离，衡量真实与生成分布相似度）
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
        logger.info(msg)
    

    if accelerator is None or accelerator.is_main_process:
        if fid < best_fid : 
            msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            logger.info(msg)
            best_fid, best_iter = fid, nb_iter
            if save:
                torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
        
        if matching_score_pred < best_matching : 
            msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
            logger.info(msg)
            best_matching = matching_score_pred

        if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
            msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
            logger.info(msg)
            best_div = diversity

        if R_precision[0] > best_top1 : 
            msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
            logger.info(msg)
            best_top1 = R_precision[0]

        if R_precision[1] > best_top2 : 
            msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
            logger.info(msg)
            best_top2 = R_precision[1]
        
        if R_precision[2] > best_top3 : 
            msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
            logger.info(msg)
            best_top3 = R_precision[2]

        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()        
def evaluation_qwen3vl_motionmillion_rpl(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, processor, eval_wrapper, comp_device, draw = True, save = True, savegif=False, accelerator=None) : 

    trans.eval()
    
    motion_annotation_list = []  #真实运动序列
    motion_pred_list = [] #生成运动序列
    R_precision_real = torch.tensor([0,0,0], device=comp_device)
    R_precision = torch.tensor([0,0,0], device=comp_device)
    matching_score_real = torch.tensor(0.0, device=comp_device)
    matching_score_pred = torch.tensor(0.0, device=comp_device)


    nb_sample = torch.tensor(0, device=comp_device) #统计验证集样本总数
    
    for i in range(1):
        global_cnt = 0

        #small_val_loader = itertools.islice(val_loader, 30)


        for batch in tqdm(val_loader):
            global_cnt += 1
            _, _, clip_text, _, pose, m_length, _, name = batch
            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22
            
            
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(comp_device)
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                typechoice=['tf2m','t2m']
                promptype = random.choice(typechoice)
                if promptype == 'tf2m':
                    frame = str(m_length[k])
                    input_template = "Give me a motion that lasts for approximately <Frame_Placeholder> frames. The caption is: <Caption_Placeholder>"
                    prompt = input_template.replace("<Caption_Placeholder>", clip_text[k]).replace("<Frame_Placeholder>",frame)
                elif promptype == 't2m':
                    input_template = "Generate a motion that reflects the text: <Caption_Placeholder>"
                    prompt = input_template.replace("<Caption_Placeholder>", clip_text[k])
                else:
                    pass  
                
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}]  
                    }
                ]
            
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                model_inputs = processor(
                    text=[text],
                    return_tensors="pt"
                ).to(trans.device)

                with torch.no_grad():
                    generated_ids = trans.generate(
                    **model_inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=processor.tokenizer.eos_token_id,  # 改用processor的tokenizer
                    eos_token_id=processor.tokenizer.eos_token_id
                )
                
                BASE_ID = 151642  
                M_CODEBOOK_SIZE = net.vqvae.quantizer.codebook_size
                
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                #breakpoint()
                if isinstance(output_ids, torch.Tensor):
                    output_ids = output_ids.cpu().tolist()
    

                valid_target_ids = [
                        tid for tid in output_ids 
                        if BASE_ID - M_CODEBOOK_SIZE - 1 <= tid <= BASE_ID  # 核心筛选条件：合法目标id区间
                    ]
                    
                    # 步骤2：反向映射还原原始motion id（核心公式：motion_id = BASE_ID - target_id）
                original_motion_ids = [
                        BASE_ID - tid for tid in valid_target_ids
                    ]
                
                all_indices  = original_motion_ids
                
                try:
                    start_token = M_CODEBOOK_SIZE
                    end_token = start_token + 1
                    start_idx = all_indices.index(start_token)
                    end_idx = all_indices.index(end_token, start_idx + 1)  # 从 start 后找 end
                    index_motion = all_indices[start_idx + 1 : end_idx]
                    index_motion = torch.tensor(index_motion, device=comp_device).unsqueeze(0)
                    pred_pose = net.forward_decoder(index_motion)

                    cur_len = pred_pose.shape[1]

                    pred_len[k] = min(cur_len, seq)
                    pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
                except:
                    continue
            #text\motion的embedding
            et_pred, em_pred = eval_wrapper.get_co_embeddings(clip_text, torch.from_numpy(val_loader.dataset.inv_transform(pred_pose_eval.detach().cpu().numpy())).to(comp_device), pred_len)

            
            if i == 0:
                pose = pose.to(comp_device).float()
                
                et, em = eval_wrapper.get_co_embeddings(clip_text, torch.from_numpy(val_loader.dataset.inv_transform(pose.detach().cpu().numpy())).to(comp_device), m_length)

                motion_annotation_list.append(em.cpu())
                motion_pred_list.append(em_pred.cpu())

                if accelerator is None or accelerator.is_main_process:
                    temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                    R_precision_real += torch.tensor(temp_R, device=comp_device)
                    matching_score_real += torch.tensor(temp_match, device=comp_device)
                    temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                    R_precision += torch.tensor(temp_R, device=comp_device)
                    matching_score_pred += torch.tensor(temp_match, device=comp_device)

                    nb_sample += et.shape[0]
                    
    if accelerator is not None:
        accelerator.wait_for_everyone()
        #聚合多卡的运动嵌入
        motion_annotation_list = accelerator.gather(torch.cat(motion_annotation_list, dim=0))
        motion_pred_list = accelerator.gather(torch.cat(motion_pred_list, dim=0))
        # reduce 多卡指标求和
        R_precision_real = accelerator.reduce(R_precision_real, reduction="sum")
        R_precision = accelerator.reduce(R_precision, reduction="sum")
        matching_score_real = accelerator.reduce(matching_score_real, reduction="sum")
        matching_score_pred = accelerator.reduce(matching_score_pred, reduction="sum")
        print(nb_sample)
        nb_sample = accelerator.reduce(nb_sample, reduction="sum")
        print(nb_sample)
    else:
        motion_annotation_list = torch.cat(motion_annotation_list, dim=0)
        motion_pred_list = torch.cat(motion_pred_list, dim=0)

    if accelerator is None or accelerator.is_main_process:
        print(len(motion_annotation_list)) # 23 ----> 48
        motion_annotation_np = motion_annotation_list.cpu().numpy()
        print(motion_annotation_np.shape) # 1472 / 1536 ---> 1530
        print(nb_sample)
        motion_pred_np = motion_pred_list.cpu().numpy()
        # 计算运动嵌入的均值（gt_mu）和协方差（gt_cov）
        gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
        mu, cov= calculate_activation_statistics(motion_pred_np)

        diversity_real = 0.0
        diversity = 0.0

        R_precision_real = R_precision_real / nb_sample
        R_precision = R_precision / nb_sample

        matching_score_real = matching_score_real / nb_sample
        matching_score_pred = matching_score_pred / nb_sample
        #计算FID（弗雷歇距离，衡量真实与生成分布相似度）
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
        logger.info(msg)
    

    if accelerator is None or accelerator.is_main_process:
        if fid < best_fid : 
            msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            logger.info(msg)
            best_fid, best_iter = fid, nb_iter
            if save:
                torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
        
        if matching_score_pred < best_matching : 
            msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
            logger.info(msg)
            best_matching = matching_score_pred

        if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
            msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
            logger.info(msg)
            best_div = diversity

        if R_precision[0] > best_top1 : 
            msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
            logger.info(msg)
            best_top1 = R_precision[0]

        if R_precision[1] > best_top2 : 
            msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
            logger.info(msg)
            best_top2 = R_precision[1]
        
        if R_precision[2] > best_top3 : 
            msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
            logger.info(msg)
            best_top3 = R_precision[2]

        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger



@torch.no_grad()        
def evaluation_transforme_root(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, root_cond_prob, text_cond_prob, force_mask_text, force_mask_root, comp_device, draw = True, save = True, savegif=False) : 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for i in range(1):
        for batch in tqdm(val_loader):
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
            root_motion = pose[:, :, :4].to(comp_device).float()
            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22
            
            text = clip.tokenize(clip_text, truncate=True).to(comp_device)

            feat_clip_text = clip_model.encode_text(text).float()
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(comp_device)
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                # try:
                index_motion = trans.sample(feat_clip_text[k:k+1], root_motion[k].unsqueeze(0), root_cond_prob, text_cond_prob, force_mask_text, force_mask_root, False)
                # except:
                #     index_motion = torch.ones(1,1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if draw:
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().to(comp_device), num_joints)

                    if i == 0 and k < 4:
                        draw_pred.append(pred_xyz)
                        draw_text_pred.append(clip_text[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            
            if i == 0:
                pose = pose.to(comp_device).float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().to(comp_device), num_joints)


                    for j in range(min(4, bs)):
                        draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                        draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample


    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()        
def evaluation_transformer_test(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, comp_deivce, draw = True, save = True, savegif=False, savenpy=False) : 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    
    for batch in val_loader:

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).to(comp_deivce)

        feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []
        for i in range(30):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).to(comp_deivce)
            pred_len = torch.ones(bs).long()
            
            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k:k+1], True)
                except:
                    index_motion = torch.ones(1,1).to(comp_deivce).long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().to(comp_deivce), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                pose = pose.to(comp_deivce).float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().to(comp_deivce), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)
    
    
    if draw:
        for ii in range(len(draw_org)):
            tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)
        
            tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"
   
    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq

def calculate_acceleration(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"
    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    pred_velocity = pred_joints[1:] - pred_joints[:-1]
    pred_acceleration = pred_velocity[1:] - pred_velocity[:-1]
    pred_mean_acceleration_seq = torch.linalg.norm(pred_acceleration, dim=-1).mean(-1) # num_poses
    pred_max_acceleration_seq = torch.linalg.norm(pred_acceleration, dim=-1).max(-1)[0] # num_poses
    
    gt_velocity = gt_joints[1:] - gt_joints[:-1]
    gt_acceleration = gt_velocity[1:] - gt_velocity[:-1]
    gt_mean_acceleration_seq = torch.linalg.norm(gt_acceleration, dim=-1).mean(-1) # num_poses
    gt_max_acceleration_seq = torch.linalg.norm(gt_acceleration, dim=-1).max(-1)[0] # num_poses

    return pred_mean_acceleration_seq, pred_max_acceleration_seq, gt_mean_acceleration_seq, gt_max_acceleration_seq

def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist