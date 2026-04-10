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

warnings.filterwarnings('ignore')
import json 

def extract_motion_ids(s):
    # 使用正则表达式提取所有数字
    ids = list(map(int, re.findall(r'<motion_id_(\d+)>', s)))
    
    # 移除第一个和最后一个元素
    if len(ids) >= 2:
        return ids[1:-1]
    return []

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



def load_model(qwen_model_path, lora_path, comp_device,args):
    
    processor = AutoProcessor.from_pretrained(
        qwen_model_path
    )

    model = AutoModelForImageTextToText.from_pretrained(
        qwen_model_path,
        dtype="auto", 
        device_map="auto"
    ).to(comp_device)

    
    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(
            model, 
            lora_path,
            device_map={"": comp_device}  
        )
        print(f"成功加载LoRA适配器:{lora_path}")
    else:
        print("未加载LoRA适配器(路径为空或不存在)")

    model.eval()  
    model.to(comp_device)
    
    print(f"Qwen3-VL with LoRA loaded successfully on {comp_device}")
        
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
    
    ckpt = torch.load(args.resume_pth, map_location='cpu')["net"]
    # net.load_state_dict(ckpt['net'], strict=True)
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    net.load_state_dict(ckpt, strict=True)
    net.eval()
    net.to(comp_device)
    print('Load VQVAE model successfully!')
    
    return processor, model, net

def chat_with_qwen(text, images, model, processor, comp_device):
    # 构建对话
    content = []
    
    # 添加文本
    if text:
        content.append({"type": "text", "text": text})
    
    # 添加图片
    for img_path in images:
        # from PIL import Image
        # try:
        #     img = Image.open(img_path).convert('RGB')
        content.append({"type": "image", "image": img_path})

    
    messages = [{"role": "user", "content": content}]
    #breakpoint()
    # 使用 processor 处理
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    # 解码
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    #breakpoint()
    return output_text[0]

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

def batch_visualize_npy_files(input_folder, output_root="vis_result"):
    """
    批量可视化指定文件夹下的所有npy文件
    
    Args:
        input_folder: 包含npy文件的文件夹路径
        output_root: 输出GIF文件的根目录，默认为 vis_result
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_root, exist_ok=True)
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(input_folder):
        # 只处理.npy文件
        if filename.endswith('.npy'):
            # 构建完整的文件路径
            npy_path = os.path.join(input_folder, filename)
            
            try:
                # 加载npy文件
                gtpose = np.load(npy_path) 
                
                # 获取文件名（不带后缀）作为基础名称
                file_base_name = os.path.splitext(filename)[0]
                
                # 恢复姿态数据
                gt_positions_with_heading = recover_from_local_rotation(gtpose.squeeze(), 22)
                
                # 构建输出GIF路径（原文件名 + .gif）
                output_path_gt = os.path.join(output_root, f'{file_base_name}.gif')
                
                # 生成可视化GIF
                visualize_smplx_85(
                    gt_positions_with_heading, 
                    output_path=output_path_gt, 
                    title=file_base_name, 
                    fps=30
                )
                
                print(f"成功处理: {filename} -> {file_base_name}.gif")
                
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")
                continue


if __name__ == "__main__":
   
    # gtpath = '/gemini-2/space/zjk/csq/project/finetrain/data/processed_data/session_194646_6/session_194646_6.npy'
    # gtpose = np.load(gtpath) 
    # gtname = "npy"
    # gt_positions_with_heading = recover_from_local_rotation(gtpose.squeeze(), 22)
    # output_path_gt = os.path.join('vis_result', f'{gtname}.gif')
    # visualize_smplx_85(gt_positions_with_heading, output_path=output_path_gt, title=gtname, fps=30)
    input_folder_path = 'vis_result/ft_4096'
    batch_visualize_npy_files(input_folder_path,input_folder_path)
    
 
   