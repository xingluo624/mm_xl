import os 
import torch
import numpy as np
import warnings
import models.vqvae as vqvae
from utils.quaternion import *

warnings.filterwarnings('ignore')


if __name__ == "__main__":
    
    data_root = '/gemini-2/space/zjk/csq/project/finetrain/data/507508/processed_data'
    comp_device = torch.device('cuda')
    splits=[]
    
    for folder_name in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder_name)

        if not os.path.isdir(folder_path):
            continue

        mp4_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".mp4")]
        mp4_path = [os.path.join(folder_path,f) for f in mp4_files]

        if len(mp4_files) == 3:
            splits.append(folder_name)
            
    print(f"Found {len(splits)} valid samples with 3 MP4 files each.")
    print(splits)

    save_txt_path = "fsqmotion_results507508.txt"
    for i in range(len(splits)):
        split = splits[i]
        
        
        # if args.motion_type == 'vector_272':
        #     motion_path = os.path.join(data_root,split,'fsq_motion_272.npy')
        # elif args.motion_type == 'vector_274':
        motion_path = os.path.join(data_root,split,'fsq_motion_274.npy')
            
        if not os.path.exists(motion_path):
            print(f'cant read {motion_path}')
            continue
       
        fsq_ids = np.load(motion_path)
        fsq_ids = fsq_ids.reshape(-1).tolist()

        
        fsqmotion = torch.tensor([fsq_ids]).to(comp_device).reshape(-1)
        print(f"真实fsq: {fsqmotion}")
        with open(save_txt_path, 'a', encoding='utf-8') as f:
            f.write(f"========== Sample: {split} ==========\n")
            f.write(f"{fsqmotion}\n\n")

        
    print('All done!')