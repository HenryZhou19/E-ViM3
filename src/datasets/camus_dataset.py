import csv
import os
from glob import glob

import numpy as np
import torch
import torchvision
from torchvision import transforms as TT

from src.utils.misc import *

from .modules.data_module_base import DataModuleBase, data_module_register

TRAIN_MEAN = 0.19323353
TRAIN_STD = 0.22046535

EF_MEAN = {
    'dummy': 0.000,
    'all': 44.412,
}

EF_STD = {
    'dummy': 1.000,
    'all': 11.845
}

EF_MIN = {
    'all': 5.000,
}

EF_MAX = {
    'all': 81.000,
}

__choice = 'dummy'
_EF_MEAN = EF_MEAN[__choice]
_EF_STD = EF_STD[__choice]
_EF_MEAN_OFFSET = EF_MEAN['all'] - _EF_MEAN
assert _EF_MEAN == 0.000 and _EF_STD == 1.000

class CAMUSDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, csv_path, split, mean=TRAIN_MEAN, std=TRAIN_STD, length=None, resize_to=None, quality_list=['Good', 'Medium', 'Poor']):
        self.root = root
        self.csv_path = csv_path
        self.split = split
                
        self.all_video_name_list = glob(self.root + '/*.npy')
        self.this_fold_video_name_list = []
        self.this_fold_video_array_list = []
        self.ef_label_list = []
        
        # these are just for debugging
        self.length = length
        self.resize_to = resize_to
        
        # print(self.all_video_name_list)
        # print(len(self.all_video_name_list))
        # print(self.csv_path)
        
        with open(self.csv_path) as csv_file:
            reader = csv.DictReader(csv_file, delimiter=',')
            
            for row in reader:  # file_name, ED, ES, NbFrame, Sex, Age, ImageQuality, EF, FrameRate, Split
                if row['Split'] == self.split:
                    if row['ImageQuality'] in quality_list:
                        self.this_fold_video_name_list.append(row['file_name'])
                        # print(os.path.join(self.root, row['name'] + '.npy'))
                        self.this_fold_video_array_list.append(np.load(os.path.join(self.root, row['file_name'] + '.npy')))
                        
                        ef_label = float(row['EF'])
                        self.ef_label_list.append(ef_label)
        
        # if split != 'train':
        #     assert mean is not None and std is not None
        #     self.mean, self.std = mean
        # else:
        #     self.mean, self.std = self.get_mean_std(mean, std)
        self.mean, self.std = self.get_mean_std(mean, std)
        print('using mean: ', self.mean)
        print('using std: ', self.std)
        
    def get_mean_std(self, mean, std):
        array_stack = np.stack(self.this_fold_video_array_list)
        print(f'{self.split}: ', array_stack.shape)
        if mean is not None and std is not None:
            return mean, std
        else:
            assert self.split == 'train'
            mean = np.mean(array_stack)
            std = np.std(array_stack)
            return mean, std

    def __getitem__(self, index):
        video_name = self.this_fold_video_name_list[index]
        video_array = self.this_fold_video_array_list[index]
        
        clip = torch.tensor(video_array, dtype=torch.float32)  # [L, C, H, W]
        
        if self.length is not None:
            assert clip.shape[0] >= self.length
            clip = clip[:self.length]
        if self.resize_to is not None:
            assert clip.shape[2] >= self.resize_to and clip.shape[3] >= self.resize_to
            clip = clip[:, :, :self.resize_to, :self.resize_to]

        if self.split == 'train':
            transform = TT.Compose([
                TT.RandomAffine(
                    degrees=10,
                    scale=(0.8, 1.1),
                    translate=(0.1, 0.1),
                    ),
                TT.Normalize([self.mean], [self.std], inplace=True),
            ])
        else:
            transform = TT.Compose([
                TT.Normalize([self.mean], [self.std], inplace=True),
            ])
        clip = transform(clip)

        clip = clip.permute(1, 0, 2, 3)  # [C, L, H, W]
        
        ef_label = (self.ef_label_list[index] - _EF_MEAN) / _EF_STD
        
        return {
            'inputs': {
                'x': torch.as_tensor(clip),
            },
            'targets': {
                'gt_ef': torch.as_tensor(ef_label, dtype=torch.float),
                'file_name': video_name,  # no file suffix
                'index': torch.tensor(index),
                # 'video_frame_real_mask': torch.as_tensor(video_frame_real_mask),
            }
        }

    def __len__(self):
        return len(self.this_fold_video_name_list)


@data_module_register('camus')
class CAMUSDataModule(DataModuleBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # XXX: use train mean and std for val and test
        self.data_dir = cfg.data.data_dir
        self.csv_path = cfg.data.csv_path
        
        # these are just for debugging
        self.length = getattr(cfg.data, 'length', None)
        self.resize_to = getattr(cfg.data, 'resize_to', None)
        self.test_quality_list = getattr(cfg.data, 'test_quality_list', ['Good', 'Medium', 'Poor'])
        
    def build_train_dataset(self):
        return CAMUSDataset(root=self.data_dir, csv_path=self.csv_path, split='train', length=self.length, resize_to=self.resize_to)
        
    def build_val_dataset(self):
        return CAMUSDataset(root=self.data_dir, csv_path=self.csv_path, split='val', length=self.length, resize_to=self.resize_to)
        
    def build_test_dataset(self):            
        return CAMUSDataset(root=self.data_dir, csv_path=self.csv_path, split='test', length=self.length, resize_to=self.resize_to, quality_list=self.test_quality_list)
    
