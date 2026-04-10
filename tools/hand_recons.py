import os
import numpy as np
import torch
import smplx

# ================= 配置区域 =================
# 请修改为你的实际路径
DATA_DIR = 'vis_result/326'  # 存放 gt_0.npz, pred_0.npz 的文件夹
SMPL_MODEL_PATH = 'body_models/human_model_files' # SMPL 模型文件夹 (包含 SMPL_MALE.pkl 等)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_SAMPLES = 40  # 0 到 9
# ===========================================

def load_smpl_model(model_path, gender='male'):
    """加载 SMPL 模型"""
    model = smplx.create(
        model_path=model_path,
        model_type='smpl',
        gender=gender,
        ext='pkl',
        num_betas=10,
        use_pca=False,
        flat_hand_mean=False
    ).to(DEVICE)
    return model

def load_npz_data(path):
    """加载 npz 数据并转为 Tensor"""
    data = np.load(path, allow_pickle=True)
    
    # 根据你提供的保存代码，键名应该是 poses, trans, betas
    # poses: (N, 24, 3), trans: (N, 3), betas: (10,)
    
    poses = torch.from_numpy(data['poses']).float().to(DEVICE)   # (N, 24, 3)
    trans = torch.from_numpy(data['trans']).float().to(DEVICE)   # (N, 3)
    betas = torch.from_numpy(data['betas']).float().to(DEVICE)   # (10,)
    
    # 扩展 betas 以匹配 batch size
    n_frames = poses.shape[0]
    betas = betas.unsqueeze(0).expand(n_frames, -1)
    
    return poses, trans, betas

def calculate_hand_error(gt_path, pred_path, smpl_model):
    """计算单个样本的手部误差"""
    # 1. 加载数据
    gt_poses, gt_trans, gt_betas = load_npz_data(gt_path)
    pred_poses, pred_trans, pred_betas = load_npz_data(pred_path)
    
    # 确保帧数一致
    min_frames = min(gt_poses.shape[0], pred_poses.shape[0])
    gt_poses, gt_trans, gt_betas = gt_poses[:min_frames], gt_trans[:min_frames], gt_betas[:min_frames]
    pred_poses, pred_trans, pred_betas = pred_poses[:min_frames], pred_trans[:min_frames], pred_betas[:min_frames]
    
    # 2. 前向推理获取关节点 (Joints)
    # SMPL 输出：vertices (N, 6890, 3), joints (N, 24, 3)
    with torch.no_grad():
        gt_output = smpl_model(body_pose=gt_poses[:, 1:],  # SMPL 输入通常把 root 分开，但 smplx 库通常接受 full pose
                               global_orient=gt_poses[:, 0:1], 
                               transl=gt_trans,
                               betas=gt_betas)
        
        pred_output = smpl_model(body_pose=pred_poses[:, 1:], 
                                 global_orient=pred_poses[:, 0:1], 
                                 transl=pred_trans,
                                 betas=pred_betas)
        
        # 获取关节点 (N, 24, 3)
        gt_joints = gt_output.joints 
        pred_joints = pred_output.joints 

    # 3. 提取手部关节 (SMPL 定义：20=左手腕，21=右手腕)
    # 如果你想算指尖，需要提取特定顶点，但 SMPL 指尖不准，建议用手腕
    gt_left_wrist = gt_joints[:, 20, :]   # (N, 3)
    gt_right_wrist = gt_joints[:, 21, :]
    
    pred_left_wrist = pred_joints[:, 20, :]
    pred_right_wrist = pred_joints[:, 21, :]
    
    # 4. 计算误差 (L2 距离)
    # 方案 A: 全局误差 (Global Error) - 包含根节点平移误差
    err_left_global = torch.norm(gt_left_wrist - pred_left_wrist, dim=1)
    err_right_global = torch.norm(gt_right_wrist - pred_right_wrist, dim=1)
    
    # 方案 B: 根节点对齐误差 (Root-Aligned) - 消除整体位移漂移，只看动作
    # 以骨盆 (Joint 0) 为基准
    gt_root = gt_joints[:, 0, :]
    pred_root = pred_joints[:, 0, :]
    
    gt_left_aligned = gt_left_wrist - gt_root
    pred_left_aligned = pred_left_wrist - pred_root
    err_left_aligned = torch.norm(gt_left_aligned - pred_left_aligned, dim=1)
    
    gt_right_aligned = gt_right_wrist - gt_root
    pred_right_aligned = pred_right_wrist - pred_root
    err_right_aligned = torch.norm(gt_right_aligned - pred_right_aligned, dim=1)
    
    return {
        'left_global': err_left_global.cpu().numpy(),
        'right_global': err_right_global.cpu().numpy(),
        'left_aligned': err_left_aligned.cpu().numpy(),
        'right_aligned': err_right_aligned.cpu().numpy(),
        'frames': min_frames
    }

def main():
    if not os.path.exists(SMPL_MODEL_PATH):
        print(f"❌ SMPL 模型路径不存在：{SMPL_MODEL_PATH}")
        print("请下载 SMPL 模型 (SMPL_MALE.pkl) 并修改配置中的 SMPL_MODEL_PATH")
        return

    print(f"🚀 开始加载 SMPL 模型 ({DEVICE})...")
    smpl_model = load_smpl_model(SMPL_MODEL_PATH, gender='male')
    
    all_left_global = []
    all_right_global = []
    all_left_aligned = []
    all_right_aligned = []
    
    print(f"📂 正在处理 {NUM_SAMPLES} 个样本...")
    
    for i in range(NUM_SAMPLES):
        gt_path = os.path.join(DATA_DIR, f'gt_{i}.npz')
        pred_path = os.path.join(DATA_DIR, f'pred_{i}.npz')
        
        if not os.path.exists(gt_path) or not os.path.exists(pred_path):
            print(f"⚠️ 跳过样本 {i} (文件缺失)")
            continue
            
        res = calculate_hand_error(gt_path, pred_path, smpl_model)
        
        all_left_global.extend(res['left_global'])
        all_right_global.extend(res['right_global'])
        all_left_aligned.extend(res['left_aligned'])
        all_right_aligned.extend(res['right_aligned'])
        
        print(f"✅ 样本 {i} 处理完成 (帧数：{res['frames']})")
        
    # 5. 统计结果 (单位：米 -> 转换为 厘米)
    to_cm = 100.0
    print("\n" + "="*40)
    print("📊 手部位置精度评估结果 (单位：厘米)")
    print("="*40)
    
    def print_metric(name, data):
        if len(data) == 0: return
        data = np.array(data) * to_cm
        print(f"{name:20s}: Mean={np.mean(data):6.2f} cm, Median={np.median(data):6.2f} cm, Std={np.std(data):6.2f}")
    
    print_metric("Left Wrist (Global)", all_left_global)
    print_metric("Right Wrist (Global)", all_right_global)
    print_metric("Left Wrist (Aligned)", all_left_aligned)
    print_metric("Right Wrist (Aligned)", all_right_aligned)
    
    # 综合手部误差
    all_hands_global = all_left_global + all_right_global
    all_hands_aligned = all_left_aligned + all_right_aligned
    print("-" * 40)
    print_metric("Avg Hand (Global)", all_hands_global)
    print_metric("Avg Hand (Aligned)", all_hands_aligned)
    print("="*40)

if __name__ == '__main__':
    main()