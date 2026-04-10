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
import time

from visualize.plot_3d_global import plot_3d_motion
from visualize.smplx2joints import process_smplx_data
#from visualize.motion_ik import convert_motion_mp4
from PIL import Image
import cv2
import imageio
from utils.face_z_align_util import rotation_6d_to_matrix, matrix_to_axis_angle
import moviepy as mp
import re
warnings.filterwarnings('ignore')

def id_to_token(motion_id):
    return f'<motion_id_{motion_id}>'

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

def parse_img_data(mp4_paths, idx):
        """
        Args:
            mp4_paths: MP4 file paths
            idx: Current frame index

        Returns:
            frames: (img_history_size, H, W, 3) image frames
        """
        # cap = cv2.VideoCapture(str(mp4_path))
        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        try:
            for mp4_path in mp4_paths:
                # decoder = VideoDecoder(mp4_path, device='cpu', dimension_order='NHWC')
                # total_frames = len(decoder)
                # if idx < total_frames:
                #     frame = decoder[idx]
                #     if frame is not None:
                #         # BGR to RGB
                #         frame = frame.cpu().numpy()
                #         frames.append(frame)
                #     else:
                #         print(f"Warning: Not enough frames in {mp4_path}")
                #         break
                # else:
                #     # If frame index exceeds total frames, use last valid frame
                #     print(f"Warning: Frame index exceeds total frames in {mp4_path}")
                #     break
                cap = cv2.VideoCapture(mp4_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
                if idx < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        frame = frame[..., ::-1]
                        #print(frame.shape)
                        frames.append(frame)
                    else:
                        print(f"Warning: Not enough frames in {mp4_path}")
                        break
                else:
                    print(f"Warning: Frame index exceeds total frames in {mp4_path}")
                    break
        except Exception as e:
            raise ValueError(f"Error loading image frames: {e}")

        # Convert to numpy array
        # frames = np.array(frames) # type: ignore

        return frames


if __name__ == "__main__":
    qwen_model_path = "/gemini-2/space/zjk/csq/mocap274_4375_326331_epoch4k"
    random_count = 40
    lora_path = None
    comp_device = torch.device('cuda')
    mean = np.load('mean_std/motionmillion/mean.npy')
    std = np.load('mean_std/motionmillion/std.npy')
    
        
    random.seed(42)
    expname = input('Input experiment name: ')
    os.makedirs(f'vis_result/{expname}', exist_ok=True)
    args = option_trans.get_args_parser()
    if args.motion_type == 'vector_274':
        print("update mean std")
        new_dim_mean = np.array([0.0, 0.0], dtype=np.float32)  
        new_dim_std = np.array([1.0, 1.0], dtype=np.float32)    
        mean = np.concatenate([mean, new_dim_mean], axis=0)  # shape (274,)
        std = np.concatenate([std, new_dim_std], axis=0) 
        
    print(f"Using device: {comp_device}")
    processor, model, net = load_model(qwen_model_path, lora_path, comp_device, args)
    
    data_root = '/gemini-2/space/zjk/csq/project/finetrain/data/processed_data326'
    
    splits=[]
    video_paths=[]
    for folder_name in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder_name)

        if not os.path.isdir(folder_path):
            continue

        mp4_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")]
        mp4_path = [os.path.join(folder_path,f) for f in mp4_files]

        if len(mp4_files) == 3:
            splits.append(folder_name)
            video_paths.append(mp4_path)
    splits = splits * 10 
    video_paths = video_paths * 10
    # splits = random.sample(splits,random_count)
    # video_paths = random.sample(video_paths,random_count)
    
    for i in range(random_count):
        split = splits[i]
        
        gt_path = os.path.join(data_root,split,f'{split}.npy')
        if args.motion_type == 'vector_272':
            motion_path = os.path.join(data_root,split,'fsq_motion_272.npy')
        elif args.motion_type == 'vector_274':
            motion_path = os.path.join(data_root,split,'fsq_motion_274.npy')
            
        text_path= os.path.join(data_root,split,'text.txt')
        if not os.path.exists(gt_path):
            print(f'cant read {gt_path}')
            continue
        if not os.path.exists(text_path):
            print(f'cant read {text_path}')
            continue
        if not os.path.exists(motion_path):
            print(f'cant read {motion_path}')
            continue
        gt_motion = np.load(gt_path)
        print(f"原始数据shape:{gt_motion.shape}")
        #breakpoint()
        idx_end = len(gt_motion) -  100
        
        idx = random.randrange(10,idx_end + 1, 2)
        
        task = random.choice(['ti2m','tim2m'])
        
        frames = parse_img_data(video_paths[i],idx)
        observations = [Image.fromarray(img) for img in frames]
        
        text = ''
        with open(text_path) as f:
            texts = f.readlines()
            text = random.choice(texts)
        
        fsq_ids = np.load(motion_path)
        fsq_ids = fsq_ids.reshape(-1).tolist()
        #breakpoint()
        messages = []
        if task == 'ti2m':
            template = "generate motion from caption and images <Caption_Placeholder>"
            prompt = template.replace("<Caption_Placeholder>", text)
            
            content = [{"type": "image", "image": img} for img in observations]
            print(content)
            content.append({"type": "text", "text": prompt})
            user_msg = {"role": "user", "content": content}
            messages.append([user_msg])
        elif task == 'tim2m':
            template = "generate motion from caption and images <Caption_Placeholder>"
            prompt = template.replace("<Caption_Placeholder>", text)
            
            motion_q = fsq_ids[:idx//2]
            motion_q = motion_q[-5:]
            
            motion_q_tokens=[id_to_token(x) for x in motion_q]
            ques = prompt + ''.join(motion_q_tokens)
            
            content = [{"type": "image", "image": img} for img in observations]
            content.append({"type": "text", "text": ques})
            user_msg = {"role": "user", "content": content}
            messages.append([user_msg])
            
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
            )
        inputs = inputs.to(model.device)
        
        t_generate_start = time.time()
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        t_generate_end = time.time()
        print(f"Generation time: {t_generate_end - t_generate_start:.4f}s")
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        print(f"生成token数{len(generated_ids_trimmed[0])}")
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        result = output_text[0]
        #breakpoint()
        _predmotion_ids = extract_motion_ids(result)
        predmotion_ids = fsq_ids[:idx//2]+_predmotion_ids
        predmotion = torch.tensor([predmotion_ids]).to(comp_device).reshape(-1)
        
        fsqmotion = torch.tensor([fsq_ids]).to(comp_device).reshape(-1)
        
        #breakpoint()
        predpose = net.forward_decoder(predmotion)
        fsqpose = net.forward_decoder(fsqmotion)
        
        # hand_result = predpose[0, :, -2:]
        # binary_hand_result = (hand_result > 0).float()
        #breakpoint()
        
        predpose = inv_transform(predpose.detach().cpu().numpy(), mean, std)
        fsqpose = inv_transform(fsqpose.detach().cpu().numpy(), mean, std)

        predname = 'pred_{}'.format(i) 
        gtname = "gt_{}".format(i)
        fsqname = 'fsq_{}'.format(i)
        #breakpoint()
        print(len(predpose[0]))
        print(len(fsqpose[0]))
        print(len(gt_motion))
        
    
        pred_positions_with_heading = recover_from_local_rotation(predpose.squeeze(0), 22)
        fsq_positions_with_heading = recover_from_local_rotation(fsqpose.squeeze(0), 22)
        gt_positions_with_heading = recover_from_local_rotation(gt_motion.squeeze(), 22)
        

        pred_npz_path = os.path.join('vis_result',expname,f'{predname}.npz')
        gt_npz_path = os.path.join('vis_result',expname,f'{gtname}.npz')
        
        save_smplx85_to_npz(pred_npz_path,pred_positions_with_heading)
        save_smplx85_to_npz(gt_npz_path,gt_positions_with_heading)
        
        #可视化
        # output_path_pred = os.path.join('vis_result', expname, f'{predname}.gif')
        # output_path_gt = os.path.join('vis_result', expname, f'{gtname}.gif')
        # output_path_fsq = os.path.join('vis_result',  expname, f'{fsqname}.gif')
        
        # visualize_smplx_85(pred_positions_with_heading, output_path=output_path_pred, title=predname, fps=args.fps)
        # visualize_smplx_85(fsq_positions_with_heading, output_path=output_path_fsq, title=fsqname, fps=args.fps)
        # visualize_smplx_85(gt_positions_with_heading, output_path=output_path_gt, title=gtname, fps=args.fps)
    
    # 保存输入文本
    # with open(os.path.join('visual_test', expname, 'texts.txt'), 'w') as f:
    #     for d in data:
    #         human_turn = next(c for c in d['conversations'] if c['from'] == 'human')
    #         f.write(human_turn['value'] + '\n')
    print('All done!')