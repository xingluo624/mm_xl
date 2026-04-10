import os
import json
from PIL import Image
import cv2
import random
import numpy as np
from datetime import datetime

def parse_img_data(mp4_paths, idx):
    """
    从多个视频路径中提取指定索引的帧
    
    Args:
        mp4_paths: MP4 文件路径列表
        idx: 当前帧索引

    Returns:
        frames: list of (H, W, 3) RGB 图像帧
    """
    frames = []
    for mp4_path in mp4_paths:
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            print(f"⚠️  Warning: Cannot open {mp4_path}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            cap.release()
            if ret:
                # BGR to RGB
                frame = frame[..., ::-1]
                frames.append(frame)
            else:
                print(f"⚠️  Warning: Failed to read frame {idx} from {mp4_path}")
        else:
            print(f"⚠️  Warning: Frame index {idx} exceeds total {total_frames} in {mp4_path}")
            cap.release()
            break
            
    return frames


def apply_random_augmentation(frames, aug_config=None):
    """
    对视频帧列表应用数据增强，同一批次使用相同参数
    
    Args:
        frames: 图像帧列表 [(H, W, 3), ...]
        aug_config: 可选的增强配置字典，None 则随机生成
        
    Returns:
        augmented_frames: 增强后的帧列表
        applied_config: 实际应用的增强配置（用于记录）
    """
    if not frames:
        return frames, None
    
    # 如果没有传入配置，则随机生成（确保整个批次一致）
    if aug_config is None:
        aug_type = random.choice(['rotation', 'brightness', 'contrast', 'exposure', 'none'])
        aug_config = {'type': aug_type}
        
        if aug_type == 'rotation':
            aug_config['angle'] = round(random.uniform(-15, 15), 2)
        elif aug_type == 'brightness':
            aug_config['factor'] = round(random.uniform(0.7, 1.3), 3)
        elif aug_type == 'contrast':
            aug_config['factor'] = round(random.uniform(0.7, 1.3), 3)
        elif aug_type == 'exposure':
            aug_config['gamma'] = round(random.uniform(0.7, 1.5), 3)
    
    augmented_frames = []
    aug_type = aug_config['type']
    
    for frame in frames:
        frame = np.asarray(frame, dtype=np.uint8)
        
        if aug_type == 'none':
            aug_frame = frame.copy()
            
        elif aug_type == 'rotation':
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            angle = aug_config['angle']
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug_frame = cv2.warpAffine(
                frame, M, (w, h), 
                flags=cv2.INTER_LINEAR, 
                borderMode=cv2.BORDER_CONSTANT, 
                borderValue=0
            )
                                   
        elif aug_type == 'brightness':
            factor = aug_config['factor']
            aug_frame = np.clip(
                frame.astype(np.float32) * factor, 0, 255
            ).astype(np.uint8)
            
        elif aug_type == 'contrast':
            factor = aug_config['factor']
            mean_val = np.mean(frame)
            aug_frame = np.clip(
                (frame.astype(np.float32) - mean_val) * factor + mean_val, 
                0, 255
            ).astype(np.uint8)
            
        elif aug_type == 'exposure':
            gamma = aug_config['gamma']
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255 
                for i in np.arange(0, 256)
            ]).astype(np.uint8)
            aug_frame = cv2.LUT(frame, table)
        
        augmented_frames.append(aug_frame)

    return augmented_frames, aug_config


def generate_filename(idx, frame_idx, aug_config, noise_std, source_idx):
    """
    生成包含处理信息的文件名
    
    格式: frame_{idx}_src{source_idx}_aug{type}_{params}_noise{std}_f{frame_idx}.png
    """
    if aug_config is None or aug_config['type'] == 'none':
        aug_str = "none"
    else:
        aug_type = aug_config['type']
        if aug_type == 'rotation':
            aug_str = f"rot{aug_config['angle']:+.1f}deg"
        elif aug_type == 'brightness':
            aug_str = f"bright{aug_config['factor']}x"
        elif aug_type == 'contrast':
            aug_str = f"contrast{aug_config['factor']}x"
        elif aug_type == 'exposure':
            aug_str = f"gamma{aug_config['gamma']:.2f}"
        else:
            aug_str = aug_type
    
    noise_str = f"noise{noise_std}" if noise_std > 0 else "clean"
    
    return f"frame_{idx:04d}_src{source_idx:02d}_aug{aug_str}_{noise_str}_f{frame_idx:02d}.png"


if __name__ == "__main__":
    random_count = 10
    data_root = '/gemini-2/space/zjk/csq/project/finetrain/data/data_331'
    save_dir = "noisy_frames"
    metadata_dir = os.path.join(save_dir, "metadata")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    
    # 收集数据
    splits = []
    video_paths = []
    for folder_name in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        mp4_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")]
        if len(mp4_files) == 3:  # 只处理包含3个视频片段的文件夹
            splits.append(folder_name)
            mp4_path = [os.path.join(folder_path, f) for f in mp4_files]
            video_paths.append(mp4_path)
    
    print(f"📁 找到 {len(splits)} 个有效数据组，准备处理 {min(random_count, len(splits))} 组\n")
    
    # 处理元数据记录
    all_metadata = []
    
    for i in range(min(random_count, len(splits))):
        split = splits[i]
        idx = random.randrange(10, 100, 2)
        
        # 1️⃣ 解析帧
        frames = parse_img_data(video_paths[i], idx)
        if not frames:
            print(f"❌ 跳过 {split}: 未获取到有效帧")
            continue
        
        # 2️⃣ 生成统一的增强配置（关键修复：批次内一致）
        aug_config = None  # 让函数内部随机生成
        
        # 3️⃣ 应用增强
        augmented_frames, applied_aug = apply_random_augmentation(frames, aug_config)
        
        # 4️⃣ 添加噪声并保存
        noise_std = 15  # 可调整噪声强度
        
        for src_idx, (orig_frame, aug_frame) in enumerate(zip(frames, augmented_frames)):
            # 添加高斯噪声
            noisy_frame = np.clip(
                aug_frame + np.random.normal(0, noise_std, aug_frame.shape), 
                0, 255
            ).astype(np.uint8)
            
            # 生成文件名（包含完整处理信息）
            filename = generate_filename(idx, src_idx, applied_aug, noise_std, src_idx)
            save_path = os.path.join(save_dir, filename)
            
            # 保存图像
            pil_img = Image.fromarray(noisy_frame)
            pil_img.save(save_path)
            
            # 记录元数据
            metadata = {
                'filename': filename,
                'split': split,
                'frame_index': idx,
                'source_video_index': src_idx,
                'source_video_path': video_paths[i][src_idx],
                'augmentation': applied_aug,
                'noise_std': noise_std,
                'image_shape': list(noisy_frame.shape),
                'timestamp': datetime.now().isoformat()
            }
            all_metadata.append(metadata)
            
            # 控制台日志
            aug_desc = applied_aug['type'] if applied_aug else 'none'
            if applied_aug and applied_aug['type'] != 'none':
                params = {k:v for k,v in applied_aug.items() if k != 'type'}
                print(f"✓ [{i+1}/{random_count}] {filename}")
                print(f"    🎨 Aug: {aug_desc} {params} | 🔊 Noise: σ={noise_std}")
            else:
                print(f"✓ [{i+1}/{random_count}] {filename}")
                print(f"    🎨 Aug: none | 🔊 Noise: σ={noise_std}")
    
    # 5️⃣ 保存元数据到 JSON 文件
    metadata_path = os.path.join(metadata_dir, f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 完成！共保存 {len(all_metadata)} 张图片")
    print(f"📋 处理详情已记录: {metadata_path}")
    print(f"📂 图片保存目录: {save_dir}")