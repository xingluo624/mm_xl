import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os

    
class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, motion_type, split,window_size = 64, unit_length = 4,add_hand=False, data_root=''):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name
        self.motion_type = motion_type
        self.add_hand = add_hand
        self.data_root = data_root

        if dataset_name == 'motionmillion':
            if not data_root:
                self.data_root = '/ssd/zhengjiakun/dataset/MotionMillion/MotionMillion'
            self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            mean = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'mean.npy'))
            std = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'std.npy'))
            split_file = pjoin(self.data_root, 'split', 'version1/tokenizer_96', split + '.txt')

        elif dataset_name == 'mocap':
            if not data_root:
                self.data_root = '/ssd/caoshiqin/datasets/our_mocap_data/processed_data'
            self.motion_dir = self.data_root
            self.text_dir = self.data_root
            self.joints_num = 22
            mean = np.load('mean_std/motionmillion/mean.npy')
            std = np.load('mean_std/motionmillion/std.npy')
            split_file = pjoin(self.data_root,  'splits', 'all.txt')
        else:
            raise KeyError('Dataset Does not Exists')
        
        id_list = []
        
        self.id_list = []
        
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # if debug:
        #     id_list = id_list[:1000]
            
        for name in tqdm(id_list):
            motion = np.load(pjoin(self.motion_dir, name + '.npy'))
            if motion.shape[0] < self.window_size:
                continue
            self.id_list.append(name)
        
        if self.dataset_name == 'mocap':
            self.id_list *= 1000
        #过滤不存在的文件或长度异常文件
        # self.id_list = self._filter_valid_ids(self.id_list)
        # print("过滤后文件数:{}".format(len(self.id_list)))
        
        if self.add_hand == True:
            new_dim_mean = np.array([0.0, 0.0], dtype=np.float32)  
            new_dim_std = np.array([1.0, 1.0], dtype=np.float32)   
            self.mean = np.concatenate([mean, new_dim_mean], axis=0)  # shape (274,)
            self.std = np.concatenate([std, new_dim_std], axis=0)    # shape (274,)
        else:
            self.mean = mean
            self.std = std
        
        print("Total number of motions {}".format(len(self.id_list)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, item):
        name = self.id_list[item]
        if self.motion_type == 'vector_272':
            motion = np.load(pjoin(self.motion_dir, name + '.npy'))
        elif self.motion_type == 'vector_274':
            motion = np.load(pjoin(self.motion_dir, name.split('/')[0],'motion_274.npy'))
        
        
        idx = random.randint(0, len(motion) - self.window_size)
        motion = motion[idx:idx+self.window_size]  #window_size 96
        
        if self.add_hand == True and self.motion_type == 'vector_272':
            
            add_zero = np.zeros((motion.shape[0], 2))
            add_zero = add_zero.astype(np.float32)
            #拼接原有motion和新增的随机维度，从272维扩展到274维
            motion = np.concatenate([motion, add_zero], axis=1)
            
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        motion = motion.astype(np.float32)
        
        return motion
    
    def _filter_valid_ids(self, all_ids):
        valid_ids = []
        for idx in all_ids:
            motion_path = pjoin(self.motion_dir, idx + '.npy')
            if os.path.exists(motion_path):
                motion = np.load(motion_path, allow_pickle=False)
                if len(motion) >= self.window_size:
                    valid_ids.append(idx)
        return valid_ids

    
def DATALoader(dataset_name,
               batch_size,
               motion_type,
                split, 
               num_workers = 64, #8,
               window_size = 64,
               unit_length = 4,
               add_hand = False,
               data_root = ''):
    print("num_workers: ", num_workers)
    trainSet = VQMotionDataset(dataset_name, 
                               motion_type, 
                               split,
                               window_size=window_size,
                               unit_length=unit_length,
                               add_hand=add_hand,
                               data_root=data_root)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last = True,
                                              pin_memory=True) 
    
    return train_loader, trainSet.mean, trainSet.std




def cycle(iterable):
    while True:
        for x in iterable:
            yield x
