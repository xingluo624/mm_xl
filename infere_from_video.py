import os 
import torch
import options.option_transformer as option_trans
import numpy as np
import warnings
import models.vqvae as vqvae
from transformers import AutoTokenizer, AutoModelForCausalLM ,AutoProcessor ,AutoModelForImageTextToText,Qwen3VLForConditionalGeneration
# from utils.quaternion import *
# from peft import PeftModel
import random
import time
# from visualize.plot_3d_global import plot_3d_motion
# from visualize.smplx2joints import process_smplx_data
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
    ids = list(map(int, re.findall(r'<motion_id_(\d+)>', s)))
    
    if len(ids) >= 2:
        return ids[1:-1]
    return []

def inv_transform(data, mean, std):
    return data * std + mean

def accumulate_rotations(relative_rotations):
    R_total = [relative_rotations[0]]
    
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    
    return np.array(R_total)

def rotations_matrix_to_smplx85(rotations_matrix, translation):
    
    nfrm, njoint, _, _ = rotations_matrix.shape
    axis_angle = matrix_to_axis_angle(torch.from_numpy(rotations_matrix)).numpy().reshape(nfrm, -1)
    
    smplx_85 = np.concatenate([axis_angle, np.zeros((nfrm, 6)), translation, np.zeros((nfrm, 10))], axis=-1)
    return smplx_85

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

def load_model(comp_device,args):
    
    processor = AutoProcessor.from_pretrained(
        args.qwen_model_path
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.qwen_model_path,
        cache_dir=None,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16 ,
    )

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

def save_smplx85_to_npz(output_path: str, smplx_85: np.ndarray, fps: float = 30.0):
    
    n_frames = smplx_85.shape[0]
    assert smplx_85.shape[1] == 85, "smplx_85 must be (n_frames, 85)"
    
    # 拆分85维参数
    body_pose = smplx_85[:, :72]          # 根节点平移 (x, y, z)
    body_pose_3d = body_pose.reshape(n_frames, -1, 3)
    root_translation = smplx_85[:, 72:75]               # 24个关节轴角旋转 (24×3=72维)
    betas = np.array([-0.4063, -0.2984,  0.5269,  1.6876, -0.8883,  
                        1.2175, -2.5507,  2.8040, -1.6811,  3.7751])                 # 10维形状参数
    
    np.savez(output_path,poses=body_pose_3d,trans=root_translation,betas=betas,gender='male',mocap_framerate=fps)
    print(f"Successfully saved SMPLX 85D data to {output_path}")
    
def parse_img_data(mp4_paths, idx):
        """
        Args:
            mp4_paths: MP4 file paths
            idx: Current frame index

        Returns:
            frames: (len(mp4_paths), H, W, 3) image frames
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
    
    random_count = 1000
    comp_device = torch.device('cuda')
    mean = np.load('mean_std/motionmillion/mean.npy')
    std = np.load('mean_std/motionmillion/std.npy')
    random.seed(42)
    args = option_trans.get_args_parser()
    
    os.makedirs(f'results/{args.exp_name}', exist_ok=True)
    
    print(f"Using device: {comp_device}")
    processor, model, net = load_model(comp_device, args)
    
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
    splits = splits * 50  
    video_paths = video_paths * 50
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
        print(gt_motion.shape)
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
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        t_generate_end = time.time()
        print(f"Generation time: {t_generate_end - t_generate_start:.4f}s")
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
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
        

        pred_npz_path = os.path.join('results',args.exp_name,f'{predname}.npz')
        gt_npz_path = os.path.join('results',args.exp_name,f'{gtname}.npz')
        
        save_smplx85_to_npz(pred_npz_path,pred_positions_with_heading)
        save_smplx85_to_npz(gt_npz_path,gt_positions_with_heading)