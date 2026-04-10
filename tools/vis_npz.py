import os 
import torch
import options.option_transformer as option_trans
import numpy as np
import warnings
import models.vqvae as vqvae
from transformers import AutoTokenizer, AutoModelForCausalLM ,AutoProcessor ,AutoModelForImageTextToText
from utils.quaternion import *
from peft import PeftModel
import random
import gc

from utils.quaternion import *
from visualize.plot_3d_global import plot_3d_motion
from visualize.smplx2joints import process_smplx_data
#from visualize.motion_ik import convert_motion_mp4
from PIL import Image
import imageio
from utils.face_z_align_util import rotation_6d_to_matrix, matrix_to_axis_angle
import moviepy as mp
import re
from qwen_vl_utils import process_vision_info
warnings.filterwarnings('ignore')
import json 


def rotations_matrix_to_smplx85(rotations_matrix, translation):
    
    nfrm, njoint, _, _ = rotations_matrix.shape
    axis_angle = matrix_to_axis_angle(torch.from_numpy(rotations_matrix)).numpy().reshape(nfrm, -1)
    
    smplx_85 = np.concatenate([axis_angle, np.zeros((nfrm, 6)), translation, np.zeros((nfrm, 10))], axis=-1)
    return smplx_85

def inv_transform(data, mean, std):
    return data * std + mean

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device).to(data.dtype)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device).to(data.dtype)
    
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

def accumulate_rotations(relative_rotations):
    R_total = [relative_rotations[0]]
    
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    
    return np.array(R_total)

def recover_from_local_position(final_x, njoint):
    # take positions_no_heading: local position on xz ori, no heading
    # velocities_root_xy_no_heading: to recover translation
    # global_heading_diff_rot: to recover root rotation
    
    nfrm, _ = final_x.shape
    positions_no_heading = final_x[:,8:8+3*njoint].reshape(nfrm, -1, 3) # frames, njoints * 3
    velocities_root_xy_no_heading = final_x[:,:2] # frames, 2
    global_heading_diff_rot = final_x[:,2:8] # frames, 6

    # recover global heading
    global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # add global heading to position
    positions_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:, None,:, :], njoint, axis=1), positions_no_heading[...,None]).squeeze(-1)

    # recover root translation
    # add heading to velocities_root_xy_no_heading

    velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)

    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)


    # add root translation
    positions_with_heading[:, :, 0] += root_translation[:, 0:1]
    positions_with_heading[:, :, 2] += root_translation[:, 2:]

    return positions_with_heading

def recover_from_local_rotation(final_x, njoint):
    # take rotations_matrix: 

    nfrm, _ = final_x.shape
    
    rotations_matrix = rotation_6d_to_matrix(torch.from_numpy(final_x[:,8+6*njoint:8+12*njoint]).reshape(nfrm, -1, 6)).numpy()
    global_heading_diff_rot = final_x[:,2:8]
    velocities_root_xy_no_heading = final_x[:,:2]
    positions_no_heading = final_x[:, 8:8+3*njoint].reshape(nfrm, -1, 3)
    height = positions_no_heading[:, 0, 1]

    global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # recover root rotation

    rotations_matrix[:,0,...] = np.matmul(inv_global_heading_rot, rotations_matrix[:,0,...])

    velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)
    
    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
    root_translation[:, 1] = height
    
    smplx_85 = rotations_matrix_to_smplx85(rotations_matrix, root_translation)
    return smplx_85

def smplx85_2_smplx322(smplx_no_shape_data):
    result = np.concatenate((smplx_no_shape_data[:,:66], np.zeros((smplx_no_shape_data.shape[0], 90)), np.zeros((smplx_no_shape_data.shape[0], 3)), np.zeros((smplx_no_shape_data.shape[0], 50)), np.zeros((smplx_no_shape_data.shape[0], 100)), smplx_no_shape_data[:,72:72+3], smplx_no_shape_data[:,75:]), axis=-1)
    
    return result

# def visualize_smplx_85(data, title=None, output_path='./recon_272/0_14_rot_new3.mp4', fps=60):
#     #smpl可视化
#     smplx_85_data = data
#     if len(smplx_85_data.shape) == 3:
#        smplx_85_data = np.squeeze(smplx_85_data, axis=0)
    
#     smplx_85_data = smplx85_2_smplx322(smplx_85_data)
#     vert, joints, motion, faces = process_smplx_data(smplx_85_data, norm_global_orient=False, transform=False)
    
#     xyz = joints[:, :22, :].reshape(-1, 22, 3).detach().cpu().numpy()
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
#     convert_motion_mp4(xyz, output_path)



def visualize_smplx_85(data, title=None, output_path='./recon_272/0_14_rot_new3.mp4', fps=60):
    #火柴人
    smplx_85_data = data
    if len(smplx_85_data.shape) == 3:
       smplx_85_data = np.squeeze(smplx_85_data, axis=0)
    
    smplx_85_data = smplx85_2_smplx322(smplx_85_data)
    vert, joints, motion, faces = process_smplx_data(smplx_85_data, norm_global_orient=False, transform=False)
    
    xyz = joints[:, :22, :].reshape(-1, 22, 3).detach().cpu().numpy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = plot_3d_motion([xyz, None, None])
    imageio.mimsave(output_path, np.array(img), fps=fps)
    out_video = mp.VideoFileClip(output_path)
    out_video.write_videofile(output_path.replace('.gif', '.mp4'))

@torch.no_grad()
def plot(pred_pose_denorm, dataname):
    
    pred_xyz = recover_from_local_rotation(pred_pose_denorm.squeeze(0).cpu().numpy(), njoint=22)
    img  = visualize_smplx_85(pred_xyz)
    return pred_xyz, img




        

def save_smplx85_to_npz(output_path: str, smplx_85: np.ndarray, fps: float = 30.0):
    
    n_frames = smplx_85.shape[0]
    assert smplx_85.shape[1] == 85, "smplx_85 must be (n_frames, 85)"
    
    # 拆分85维参数
    body_pose = smplx_85[:, :72]          # 根节点平移 (x, y, z)
    body_pose_3d = body_pose.reshape(n_frames, -1, 3)
    root_translation = smplx_85[:, 72:75]               # 24个关节轴角旋转 (24×3=72维)
    betas = np.array([-0.4063, -0.2984,  0.5269,  1.6876, -0.8883,  
                        1.2175, -2.5507,  2.8040, -1.6811,  3.7751])                 # 10维形状参数
    
    # 构造与图中一致的字典结构
    np.savez(output_path,poses=body_pose_3d,trans=root_translation,betas=betas,gender='male',mocap_framerate=fps)
    print(f"Successfully saved SMPLX 85D data to {output_path}")
    
if __name__ == "__main__":

    data = np.load("/gemini-2/space/zjk/csq/project/MotionMillion-Codes/vis_result/pred_1.npz")
    
    poses = data['poses']  
    trans = data['trans']  
    betas = data['betas']  

    N = poses.shape[0]  

    poses_flat = poses.reshape(N, -1)  
    betas_expand = np.tile(betas, (N, 1))  

    
    smpl85 = np.concatenate([poses_flat, trans, betas_expand], axis=1)
    #breakpoint()
    

    save_path = os.path.join('vis_result','npz.gif')
    
    visualize_smplx_85(smpl85, output_path=save_path, title='vis_npz', fps=30)
    
    
    # output_path_fsq = os.path.join('vis_result',  expname, f'{fsqname}.gif')
    # output_path_pred = os.path.join('vis_result', expname, f'{predname}.gif')
    # output_path_gt = os.path.join('vis_result', expname, f'{gtname}.gif')
    
    # visualize_smplx_85(fsq_positions_with_heading, output_path=output_path_fsq, title=fsqname, fps=args.fps)
    # visualize_smplx_85(pred_positions_with_heading, output_path=output_path_pred, title=predname, fps=args.fps)
    # visualize_smplx_85(gt_positions_with_heading, output_path=output_path_gt, title=gtname, fps=args.fps)
