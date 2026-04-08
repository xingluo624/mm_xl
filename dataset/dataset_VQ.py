import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import os


class VQMotionDatasetEval(data.Dataset):
    def __init__(self, dataset_name,  motion_type, text_type, version, split, debug, window_size = 64, unit_length = 4,add_hand=False):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name
        self.motion_type = motion_type
        self.text_type = text_type
        self.version = version
        self.add_hand = add_hand

        if dataset_name == 'motionmillion':
            self.data_root = '/ssd/zhengjiakun/dataset/MotionMillion/MotionMillion'
            self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)
            self.text_dir = pjoin(self.data_root, self.text_type)
            self.joints_num = 22
            self.max_motion_length = 300
            mean = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'mean.npy'))
            std = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'std.npy'))
            split_file = pjoin(self.data_root, 'split', self.version, split + '.txt')
        else:
            raise KeyError('Dataset Does not Exists')
        
        joints_num = self.joints_num
        id_list = []
        
        self.data = []
        self.lengths = []
        self.id_list = []
        
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        if debug:
            id_list = id_list[:1000]
            
        for name in tqdm(id_list):
            motion = np.load(pjoin(self.motion_dir, name + '.npy'))
            if motion.shape[0] < self.window_size:
                continue
            self.id_list.append(name)
            self.lengths.append(motion.shape[0] - self.window_size)
            self.data.append(motion)
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
        motion = self.data[item]
        
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
        if self.add_hand == True:
            random_dim_int = np.random.randint(0, 2, size=(motion.shape[0], 2))  
            
            random_dim = random_dim_int.astype(np.float32)
            #拼接原有motion和新增的随机维度，从272维扩展到274维
            motion = np.concatenate([motion, random_dim], axis=1)
            
        return motion, m_length, name
    
class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name,  motion_type, text_type, version, split, debug, window_size = 64, unit_length = 4,add_hand=False):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name
        self.motion_type = motion_type
        self.text_type = text_type
        self.version = version
        self.add_hand = add_hand
        
        if dataset_name == 'motionmillion':
            self.data_root = 'data/MotionMillion'
            self.motion_dir = pjoin(self.data_root, 'motion_data', self.motion_type)
            self.text_dir = pjoin(self.data_root, self.text_type)
            self.joints_num = 22
            mean = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'mean.npy'))
            std = np.load(pjoin(self.data_root, 'mean_std', self.motion_type, 'std.npy'))
            split_file = pjoin(self.data_root, 'split', self.version, split + '.txt')

        elif dataset_name == 'mocap':
            self.data_root = 'data/our_mocap_data/processed_data'
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

        if debug:
            id_list = id_list[:1000]
            
        for name in tqdm(id_list):
            self.id_list.append(name)
        #过滤不存在的文件或长度异常文件
        self.id_list = self._filter_valid_ids(self.id_list)
        print("过滤后文件数:{}".format(len(self.id_list)))
        
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
                text_type,
                version, 
                split, 
                debug,
               num_workers = 64, #8,
               window_size = 64,
               unit_length = 4,
               add_hand = False):
    print("num_workers: ", num_workers)
    trainSet = VQMotionDataset(dataset_name, motion_type, text_type, version, split, debug, window_size=window_size, unit_length=unit_length,add_hand=add_hand)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True,
                                              pin_memory=True) #,
                                            #   prefetch_factor=4)
    
    return train_loader, trainSet.mean, trainSet.std


def DATALoaderEvalVQ(dataset_name,
               batch_size,
               motion_type,
                text_type,
                version, 
                split, 
                debug,
               num_workers = 64, #8,
               window_size = 64,
               unit_length = 4,
               add_hand = False):
    print("num_workers: ", num_workers)
    trainSet = VQMotionDatasetEval(dataset_name, motion_type, text_type, version, split, debug, window_size=window_size, unit_length=unit_length,add_hand=add_hand)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True,
                                              pin_memory=True) #,
                                            #   prefetch_factor=4)
    
    return train_loader, trainSet.mean, trainSet.std


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
