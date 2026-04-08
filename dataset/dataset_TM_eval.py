import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm

import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class MotionMillionFSQDataset(data.Dataset):
    def __init__(self, dataset_name, is_test,  feat_bias = 5, max_text_len = 20, unit_length = 4, version = "version1/tokenizer_no_mirror",add_hand=False,motion_type='vector_272'):
        
        self.max_length = 20
        self.pointer = 0
        self.dataset_name = dataset_name
        self.is_test = is_test
        self.max_text_len = max_text_len
        self.unit_length = unit_length
        
        self.version = version
        self.add_hand = add_hand
        self.motion_type = motion_type
        
        if dataset_name == 'motionmillion':
            # self.data_root = './dataset/HumanML3D'
            # self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            # self.text_dir = pjoin(self.data_root, 'texts')
            self.data_root = '/ssd/zhengjiakun/dataset/MotionMillion/MotionMillion'
            self.motion_dir = pjoin(self.data_root, 'motion_data', "vector_272")
            self.text_dir = pjoin(self.data_root, "texts")
            self.joints_num = 22
            radius = 4
            fps = 60
            self.max_motion_length = 600
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
            self.meta_dir = pjoin(self.data_root, 'mean_std', "vector_272")
            if is_test:
                split_file = pjoin(self.data_root, 'split', self.version, 'test.txt')
            else:
                split_file = pjoin(self.data_root, 'split', self.version, 'val.txt')
        
        elif dataset_name == 'mocap':
            self.data_root = '/ssd/caoshiqin/datasets/our_mocap_data/processed_data'
            self.motion_dir = self.data_root
            self.text_dir = self.data_root
            self.joints_num = 22
            self.max_motion_length = 900
            self.meta_dir = '/ssd/zhengjiakun/dataset/MotionMillion/MotionMillion/mean_std/vector_272'
            split_file = pjoin(self.data_root,'splits' ,'all.txt')
            
            
            
        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        
        if self.dataset_name == 'motionmillion':
            min_motion_len = 120 # 192
        elif self.dataset_name == 't2m':
            min_motion_len = 40 # 192
        else:
            min_motion_len = 24

        id_list = []
        self.id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        length_list = []
        for name in tqdm(id_list):
            try:
                if self.motion_type == 'vector_272':
                    motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                elif self.motion_type == 'vector_274':
                    motion = np.load(pjoin(self.motion_dir, name.split('/')[0],'motion_274.npy'))
                if (len(motion)) < min_motion_len or (len(motion) > self.max_motion_length):
                    continue
                self.id_list.append(name)
                length_list.append(len(motion))
                
            except Exception as e:
                print(e)
        
        self.id_list.sort()
        
        if self.add_hand == True:
            new_dim_mean = np.array([0.0, 0.0], dtype=np.float32)  
            new_dim_std = np.array([1.0, 1.0], dtype=np.float32)   

            
            self.mean = np.concatenate([mean, new_dim_mean], axis=0)  # shape (274,)
            self.std = np.concatenate([std, new_dim_std], axis=0)    # shape (274,)
        else:
            self.mean = mean
            self.std = std
            
        self.length_arr = np.array(length_list)
        self.reset_max_len(self.max_length)
        print(len(self.id_list)) # 
    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self, data):
        return (data - self.mean) / self.std

    def __len__(self):
        return len(self.id_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        name = self.id_list[idx]
        if self.motion_type == 'vector_272':
            motion = np.load(pjoin(self.motion_dir, name + '.npy'))
        elif self.motion_type == 'vector_274':
            motion = np.load(pjoin(self.motion_dir, name.split('/')[0],'motion_274.npy'))
        
        if self.add_hand == True and self.motion_type == 'vector_272':
            add_zero = np.zeros((motion.shape[0], 2))
            add_zero = add_zero.astype(np.float32)
    
            motion = np.concatenate([motion, add_zero], axis=1)

        m_length = len(motion)
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)

        motion = motion.astype(np.float32)
        return motion, m_length, name



def MotionMillionFSQDATALoader(dataset_name, is_test,
                batch_size, w_vectorizer,
                num_workers = 8, unit_length = 4, version = "version1/tokenizer_no_mirror",add_hand=False,motion_type='vector_274') : 
    
    val_dataset = MotionMillionFSQDataset(dataset_name, is_test, unit_length=unit_length, version=version,add_hand=add_hand,motion_type=motion_type)
    val_loader = torch.utils.data.DataLoader( val_dataset, 
                                              batch_size,
                                              shuffle = True,
                                              num_workers=40, # num_workers,
                                              collate_fn=collate_fn,
                                              drop_last = True,
                                              prefetch_factor=2)
    return val_loader, val_dataset.mean, val_dataset.std

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
