import os 
import torch
import options.option_transformer as option_trans
import numpy as np
import warnings
import models.vqvae as vqvae
from transformers import AutoProcessor, AutoModelForImageTextToText
from utils.quaternion import *
from peft import PeftModel
import random
import time
from torchvision.transforms import v2
from visualize.plot_3d_global import plot_3d_motion
from visualize.smplx2joints import process_smplx_data
from PIL import Image
import cv2
import imageio
from utils.face_z_align_util import rotation_6d_to_matrix, matrix_to_axis_angle
import moviepy as mp
import re
warnings.filterwarnings('ignore')
from decord import VideoReader, cpu

# -----------------------------------------------------------------------
# Constants (must match training)
# -----------------------------------------------------------------------
HIST_TOKEN_COUNT = 5   # number of history tokens taken from end of prev chunk
CHUNK_SIZE = 64


# -----------------------------------------------------------------------
# Token helpers
# -----------------------------------------------------------------------
def id_to_token(motion_id):
    return f'<motion_id_{motion_id}>'


def extract_motion_ids(s):
    """Extract motion token ids from model output string, dropping SOM/EOM."""
    ids = list(map(int, re.findall(r'<motion_id_(\d+)>', s)))
    if len(ids) >= 2:
        return ids[1:-1]
    return []


# -----------------------------------------------------------------------
# SMPLX / rotation utilities
# -----------------------------------------------------------------------
def rotations_matrix_to_smplx85(rotations_matrix, translation):
    nfrm, njoint, _, _ = rotations_matrix.shape
    axis_angle = matrix_to_axis_angle(
        torch.from_numpy(rotations_matrix)
    ).numpy().reshape(nfrm, -1)
    smplx_85 = np.concatenate(
        [axis_angle, np.zeros((nfrm, 6)), translation, np.zeros((nfrm, 10))],
        axis=-1
    )
    return smplx_85


def inv_transform(data, mean, std):
    return data * std + mean


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device).to(data.dtype)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device).to(data.dtype)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
        positions
    )
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


def accumulate_rotations(relative_rotations):
    R_total = [relative_rotations[0]]
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    return np.array(R_total)


def recover_from_local_position(final_x, njoint):
    nfrm, _ = final_x.shape
    positions_no_heading = final_x[:, 8:8 + 3 * njoint].reshape(nfrm, -1, 3)
    velocities_root_xy_no_heading = final_x[:, :2]
    global_heading_diff_rot = final_x[:, 2:8]

    global_heading_rot = accumulate_rotations(
        rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy()
    )
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    positions_with_heading = np.matmul(
        np.repeat(inv_global_heading_rot[:, None, :, :], njoint, axis=1),
        positions_no_heading[..., None]
    ).squeeze(-1)

    velocities_root_xyz_no_heading = np.zeros(
        (velocities_root_xy_no_heading.shape[0], 3)
    )
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(
        inv_global_heading_rot[:-1],
        velocities_root_xyz_no_heading[1:, :, None]
    ).squeeze(-1)
    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)

    positions_with_heading[:, :, 0] += root_translation[:, 0:1]
    positions_with_heading[:, :, 2] += root_translation[:, 2:]
    return positions_with_heading


def recover_from_local_rotation(final_x, njoint):
    nfrm, _ = final_x.shape

    rotations_matrix = rotation_6d_to_matrix(
        torch.from_numpy(
            final_x[:, 8 + 6 * njoint:8 + 12 * njoint]
        ).reshape(nfrm, -1, 6)
    ).numpy()
    global_heading_diff_rot = final_x[:, 2:8]
    velocities_root_xy_no_heading = final_x[:, :2]
    positions_no_heading = final_x[:, 8:8 + 3 * njoint].reshape(nfrm, -1, 3)
    height = positions_no_heading[:, 0, 1]

    global_heading_rot = accumulate_rotations(
        rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy()
    )
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    rotations_matrix[:, 0, ...] = np.matmul(
        inv_global_heading_rot, rotations_matrix[:, 0, ...]
    )

    velocities_root_xyz_no_heading = np.zeros(
        (velocities_root_xy_no_heading.shape[0], 3)
    )
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(
        inv_global_heading_rot[:-1],
        velocities_root_xyz_no_heading[1:, :, None]
    ).squeeze(-1)

    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
    root_translation[:, 1] = height

    smplx_85 = rotations_matrix_to_smplx85(rotations_matrix, root_translation)
    return smplx_85


def smplx85_2_smplx322(smplx_no_shape_data):
    result = np.concatenate((
        smplx_no_shape_data[:, :66],
        np.zeros((smplx_no_shape_data.shape[0], 90)),
        np.zeros((smplx_no_shape_data.shape[0], 3)),
        np.zeros((smplx_no_shape_data.shape[0], 50)),
        np.zeros((smplx_no_shape_data.shape[0], 100)),
        smplx_no_shape_data[:, 72:72 + 3],
        smplx_no_shape_data[:, 75:]
    ), axis=-1)
    return result


def visualize_smplx_85(data, title=None, output_path='./recon_272/0_14_rot_new3.mp4', fps=60):
    smplx_85_data = data
    if len(smplx_85_data.shape) == 3:
        smplx_85_data = np.squeeze(smplx_85_data, axis=0)

    smplx_85_data = smplx85_2_smplx322(smplx_85_data)
    vert, joints, motion, faces = process_smplx_data(
        smplx_85_data, norm_global_orient=False, transform=False
    )
    xyz = joints[:, :22, :].reshape(-1, 22, 3).detach().cpu().numpy()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = plot_3d_motion([xyz, None, None])
    imageio.mimsave(output_path, np.array(img), fps=fps)
    out_video = mp.VideoFileClip(output_path)
    out_video.write_videofile(output_path.replace('.gif', '.mp4'))


# -----------------------------------------------------------------------
# Motion encoding helpers (aligned with training dataset)
# -----------------------------------------------------------------------
@torch.no_grad()
def encode_motion_chunk(net, motion_np, comp_device):
    """Encode a (T, D) numpy motion slice -> list[int] token ids."""
    pose = torch.from_numpy(motion_np).float().unsqueeze(0).to(comp_device)
    tokens = net.encode(pose)          # (1, N_tokens)
    return tokens.cpu().numpy().tolist()[0]


def get_history_tokens(net, motion, idx, comp_device):
    """
    Return the last HIST_TOKEN_COUNT token ids from the previous chunk.
    Returns None on cold start (idx < CHUNK_SIZE), matching training logic.
    """
    if idx < CHUNK_SIZE:
        return None
    prev_start = idx - CHUNK_SIZE
    prev_chunk = motion[prev_start: prev_start + CHUNK_SIZE]
    prev_ids = encode_motion_chunk(net, prev_chunk, comp_device)
    return prev_ids[-HIST_TOKEN_COUNT:]


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------
def load_model(qwen_model_path, lora_path, comp_device, args):
    processor = AutoProcessor.from_pretrained(qwen_model_path)

    model = AutoModelForImageTextToText.from_pretrained(
        qwen_model_path,
        dtype="auto",
        device_map="auto"
    )

    if lora_path and os.path.exists(lora_path):
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            device_map={"": comp_device}
        )
        print(f"成功加载LoRA适配器: {lora_path}")
    else:
        print("未加载LoRA适配器(路径为空或不存在)")

    model.eval()
    print(f"Qwen3-VL loaded successfully on {comp_device}")

    net = vqvae.HumanVQVAE(
        args,
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
        args.use_attn
    )

    ckpt = torch.load(args.resume_pth, map_location='cpu')["net"]
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    net.load_state_dict(ckpt, strict=True)
    net.eval()
    print('Load VQVAE model successfully!')

    return processor, model, net


# -----------------------------------------------------------------------
# Image frame loading
# -----------------------------------------------------------------------
def parse_img_data(mp4_paths, idx):
    """
    Args:
        mp4_paths: list of MP4 file paths
        idx:       frame index to extract
    Returns:
        frames: list of (H, W, 3) RGB numpy arrays
    """
    frames = []
    try:
        for mp4_path in mp4_paths:
            vr = VideoReader(mp4_path, ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            if idx < total_frames:
                frame = vr[idx].asnumpy()
                frames.append(frame)
            else:
                print(f"Warning: Frame index {idx} exceeds total frames {total_frames} in {mp4_path}")
                break
            del vr
    except Exception as e:
        raise ValueError(f"Error loading image frames: {e}")
    return frames


# -----------------------------------------------------------------------
# NPZ saving utility
# -----------------------------------------------------------------------
def save_smplx85_to_npz(output_path, smplx_85, fps=30.0):
    n_frames = smplx_85.shape[0]
    assert smplx_85.shape[1] == 85, "smplx_85 must be (n_frames, 85)"

    body_pose = smplx_85[:, :72]
    body_pose_3d = body_pose.reshape(n_frames, -1, 3)
    root_translation = smplx_85[:, 72:75]
    betas = np.array([
        -0.4063, -0.2984,  0.5269,  1.6876, -0.8883,
         1.2175, -2.5507,  2.8040, -1.6811,  3.7751
    ])
    np.savez(
        output_path,
        poses=body_pose_3d,
        trans=root_translation,
        betas=betas,
        gender='male',
        mocap_framerate=fps
    )
    print(f"Saved SMPLX 85D data to {output_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == "__main__":
    qwen_model_path = "/data_public/zjk/csq/PyProject/ft_qwenvl/logs/507508_2_1m_chunk_epoch16k/checkpoint-66000"
    random_count = 40
    lora_path = None
    comp_device = torch.device('cuda')

    mean = np.load('mean_std/motionmillion/mean.npy')
    std  = np.load('mean_std/motionmillion/std.npy')

    expname = input('Input experiment name: ')
    os.makedirs(f'vis_result/{expname}', exist_ok=True)

    args = option_trans.get_args_parser()

    if args.motion_type == 'vector_274':
        print("update mean std")
        new_dim_mean = np.array([0.0, 0.0], dtype=np.float32)
        new_dim_std  = np.array([1.0, 1.0], dtype=np.float32)
        mean = np.concatenate([mean, new_dim_mean], axis=0)
        std  = np.concatenate([std,  new_dim_std],  axis=0)

    print(f"Using device: {comp_device}")
    processor, model, net = load_model(qwen_model_path, lora_path, comp_device, args)

    # ----------------------------------------------------------------
    # Build sample list
    # ----------------------------------------------------------------
    data_root = '/data_public/zjk/csq/PyProject/ft_qwenvl/data/507508_2_1m/processed_data'

    splits      = []
    video_paths = []
    for folder_name in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        mp4_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")]
        mp4_path  = sorted(os.path.join(folder_path, f) for f in mp4_files)
        if len(mp4_files) == 3:
            splits.append(folder_name)
            video_paths.append(mp4_path)

    print(f"Found {len(splits)} valid samples with 3 MP4 files each.")

    resize_transform = v2.Compose([
        v2.Resize(size=(270, 480), interpolation=v2.InterpolationMode.BILINEAR)
    ])

    # ----------------------------------------------------------------
    # Inference loop
    # ----------------------------------------------------------------
    for i in range(min(random_count, len(splits))):
        split = splits[i]
        print(f"\n{'='*60}")
        print(f"[Sample {i}]  split = {split}")

        # --- paths ---
        if args.motion_type == 'vector_272':
            motion_path = os.path.join(data_root, split, 'motion_272.npy')
        elif args.motion_type == 'vector_274':
            motion_path = os.path.join(data_root, split, 'motion_274.npy')
        else:
            motion_path = os.path.join(data_root, split, 'motion_272.npy')

        text_path = os.path.join(data_root, split, 'text.txt')

        if not os.path.exists(text_path):
            print(f'  [SKIP] cant read {text_path}')
            continue
        if not os.path.exists(motion_path):
            print(f'  [SKIP] cant read {motion_path}')
            continue

        # --- load & normalise motion ---
        gt_motion_raw = np.load(motion_path)
        print(f'  motion shape: {gt_motion_raw.shape}')
        gt_motion = (gt_motion_raw - mean) / std

        # --- sample idx (same strategy as training: step=2) ---
        idx_end = len(gt_motion) - CHUNK_SIZE
        idx = random.randrange(0, idx_end + 1, 2)
        has_history = idx >= CHUNK_SIZE
        print(f'  idx={idx}  has_history={has_history}')

        # ---- ground-truth token ids for this chunk ----
        pose_chunk = gt_motion[idx: idx + CHUNK_SIZE]
        true_ids = encode_motion_chunk(net, pose_chunk, comp_device)

        # ---- history tokens (aligned with training, no dropout/corruption at test time) ----
        hist_ids = get_history_tokens(net, gt_motion, idx, comp_device)
        hist_token_str = (
            ''.join(id_to_token(x) for x in hist_ids)
            if hist_ids is not None else None
        )
        print(f'  hist_ids : {hist_ids}')

        # ---- image frames ----
        frames = parse_img_data(video_paths[i], idx)
        frames = [np.array(resize_transform(Image.fromarray(img))) for img in frames]
        observations = [Image.fromarray(img) for img in frames]

        # ---- caption ----
        with open(text_path) as f:
            texts = f.readlines()
        text = random.choice(texts).strip()
        print(f'  caption  : {text}')

        # ---- build prompt (aligned with training) ----
        # training template: "generate motion from caption and images <Caption_Placeholder>"
        # + hist_token_str appended after caption when available
        template = "generate motion from caption and images <Caption_Placeholder>"
        prompt = template.replace("<Caption_Placeholder>", text)
        if hist_token_str is not None:
            prompt = prompt + hist_token_str

        content = [{"type": "image", "image": img} for img in observations]
        content.append({"type": "text", "text": prompt})
        user_msg = {"role": "user", "content": content}
        messages = [[user_msg]]

        # ---- tokenise ----
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # ---- generate ----
        t0 = time.time()
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        elapsed = time.time() - t0
        print(f'  generation time : {elapsed:.3f}s')

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        print(f'  generated tokens: {len(generated_ids_trimmed[0])}')

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        result = output_text[0]

        # ---- extract predicted ids ----
        pred_ids = extract_motion_ids(result)

        # ----------------------------------------------------------------
        # Print comparison
        # ----------------------------------------------------------------
        print(f'\n  --- token comparison ---')
        print(f'  true_ids (len={len(true_ids)}): {true_ids}')
        print(f'  pred_ids (len={len(pred_ids)}): {pred_ids}')

        min_len = min(len(true_ids), len(pred_ids))
        if min_len > 0:
            match = sum(t == p for t, p in zip(true_ids[:min_len], pred_ids[:min_len]))
            print(f'  token accuracy (first {min_len}): {match}/{min_len} = {match / min_len:.4f}')

            # per-position diff for easier debugging
            diffs = [
                f'{j}:({true_ids[j]}->{pred_ids[j]})'
                for j in range(min_len)
                if true_ids[j] != pred_ids[j]
            ]
            if diffs:
                print(f'  mismatches: {diffs[:20]}{"..." if len(diffs) > 20 else ""}')
            else:
                print(f'  perfect match on first {min_len} tokens!')
        else:
            print('  pred_ids is empty — model may not have generated valid motion tokens')

        print(f'  raw model output: {result[:200]}{"..." if len(result) > 200 else ""}')

    print('\nAll done!')