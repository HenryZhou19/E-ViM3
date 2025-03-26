import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as TT
import torchvision

from .modules.data_module_base import DataModuleBase, data_module_register
from .modules.media_rw import load_video

BUV_MEAN = [86.8737] * 3
BUV_STD = [47.8046] * 3

class VideoRecord(object):
    def __init__(self, root, row):
        self.root = root
        self._data = row

    @property
    def path(self):
        return os.path.join(self.root, self._data[0])

    @property
    def label(self):
        return int(self._data[1])
    
    @property
    def name(self):
        return os.path.split(self._data[0])[-1]


    
class BuvDataset(Dataset):
    def __init__(
        self,
        videos_root_path,
        video_name_file_path,
        transform,
        clips=1,
        length=32,
        period=4,
        is_train=False,
        ) -> None:
        super().__init__()
        
        self.videos_root_path = videos_root_path

        self.video_name_file_path = video_name_file_path

        self.video_list = self._get_video_list()  # [VideoRecord, VideoRecord, ...]

        self.clips = clips
        self.length = length
        assert self.length % self.clips == 0
        self.period = period
        
        self.transform = transform
        self.train = is_train


    def _get_video_list(self):
        # check the frame number is large >3:
        # tmp = [x.strip().split(' ') for x in open(self.video_name_file_path)]  # path, length, label
        # if not self.test_mode or self.remove_missing:
        #     tmp = [item for item in tmp if int(item[1]) >= 3]
        video_list = [VideoRecord(self.videos_root_path, x.strip().split(' ')) for x in open(self.video_name_file_path)]
        print('video number:%d' % (len(video_list)))
        return video_list
        
    
    def _get_video_clip(self, video, clip_length, period_this, start_index):
        frame_idx_fn = lambda s: s + period_this * torch.arange(clip_length)
        frame_indices = frame_idx_fn(start_index)

        video = video[:, frame_indices, :, :]
        return video
    
    
    def _get_video(self, video_record: VideoRecord):
  
        video = load_video(video_record.path, gray_out=True)
        
        video = torch.as_tensor(video, dtype=torch.float).transpose(0, 1) / 255.   # [L, C, HH, WW]
        if self.transform is not None:
            video = self.transform(video).transpose(0, 1)  # [C, L, H, W]
        
        c, f, h, w = video.shape
        video_frame_real_mask = torch.ones(f, dtype=bool)
        
        if isinstance(self.period, list):
            period_this = np.random.choice(self.period)
        else:
            period_this = self.period
        
        if f < self.length * self.period:
            # zero padding
            # video = torch.cat([video, torch.zeros((c, self.length * self.period - f, h, w), dtype=video.dtype)], dim=1)
            
            # repeat padding
            repeat_time, repeat_tail = (self.length * period_this) // f, (self.length * period_this) % f
            video = torch.cat([video] * repeat_time + [video[:, :repeat_tail, ...]], dim=1)
            
            video_frame_real_mask = torch.cat((video_frame_real_mask, torch.zeros(self.length * period_this - f, dtype=video_frame_real_mask.dtype)), dim=0)
            c, f, h, w = video.shape
        
        clips_start_index = torch.linspace(0, f, steps=self.clips + 1)
        clips_start_index = torch.round(clips_start_index).int()
        clip_length = self.length // self.clips
        video_all = []
        for i_clip in range(self.clips):
            clip_start = clips_start_index[i_clip]
            clip_end = clips_start_index[i_clip + 1]
            video_this_clip = video[:, clip_start:clip_end]
            f = clip_end - clip_start
            remainder_for_start = f - (clip_length - 1) * period_this
            if self.train:
                start_index = torch.randint(0, remainder_for_start, (1,))
            else:
                start_index = torch.tensor([remainder_for_start // 2])
            
            video_clip = self._get_video_clip(video_this_clip, clip_length, period_this, start_index)
            video_all.append(video_clip)
            # print(video_clip.shape)
        video_all = torch.cat(video_all, dim=1)
        
        return video_all
    
    def __getitem__(self, index):
        video_record = self.video_list[index]  # path, length
        
        # segment_start_indexs = self._sample_segments(video_record.num_frames, is_train=self.is_train)
        
        video = self._get_video(video_record)
        label = torch.as_tensor(video_record.label, dtype=torch.float)

        return {
            'inputs':{
                'x': video,
                # 'video_frame_real_mask': video_frame_real_mask,
            },
            'targets':{
                'y': label,
                'video_name': video_record.name,
            },
        }

    def __len__(self):
        return len(self.video_list)
    

@data_module_register('buv')
class BuvDataModule(DataModuleBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.kwargs = {
            'videos_root_path': cfg.data.data_dir,
            'clips': cfg.data.clips,
            'length': cfg.data.length,
            'period': cfg.data.period,
            }
        self.train_video_name_file_path = cfg.data.traintxt
        self.val_video_name_file_path = cfg.data.valtxt
        
        MEAN = BUV_MEAN
        STD = BUV_STD
        
        if getattr(cfg.data, 'trainset_augmentation', False):
            self.train_transform = TT.Compose([
                TT.RandomAffine(degrees=(-30, 30), translate=(0.2, 0.2), scale=(0.8, 1.2)),
                # TT.Resize((cfg.data.resize_to, cfg.data.resize_to), antialias=True),
                TT.RandomResizedCrop((cfg.data.resize_to, cfg.data.resize_to), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),
                TT.RandomHorizontalFlip(p=0.5),  # XXX
                # TT.RandomVerticalFlip(p=0.5),
                TT.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
                TT.Normalize([i / 255. for i in MEAN[:1]], [i / 255. for i in STD[:1]], inplace=True),
            ])
        else:
            self.train_transform = TT.Compose([
                TT.Resize((cfg.data.resize_to, cfg.data.resize_to), antialias=True),
                # TT.RandomHorizontalFlip(p=0.5),  # XXX
                TT.Normalize([i / 255. for i in MEAN[:1]], [i / 255. for i in STD[:1]], inplace=True),
            ])
        
        self.val_transform = TT.Compose([
            TT.Resize((cfg.data.resize_to, cfg.data.resize_to), antialias=True),
            TT.Normalize([i / 255. for i in MEAN[:1]], [i / 255. for i in STD[:1]], inplace=True),
        ])

    def build_train_dataset(self):
        return BuvDataset(**self.kwargs,
                          video_name_file_path=self.train_video_name_file_path,
                          transform=self.train_transform,
                          is_train=True,
                          )
        
    def build_val_dataset(self):
        return BuvDataset(**self.kwargs,
                          video_name_file_path=self.val_video_name_file_path,
                          transform=self.val_transform,
                          )
        
    def build_test_dataset(self):
        print('Using val dataset as test dataset')
        return self.build_val_dataset()
    
    
