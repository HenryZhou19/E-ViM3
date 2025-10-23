# ------------------------------------------------------------------------
# Modified from "Video-based AI for beat-to-beat assessment of cardiac function"
# (https://doi.org/10.1038/s41586-020-2145-8)
# (https://echonet.github.io/dynamic/)
# ------------------------------------------------------------------------


# import gzip
import os
import warnings
# import pickle
from collections import defaultdict

import cv2
import numpy as np
import pandas
import skimage.draw
import torch
import torchvision
from skimage.metrics import mean_squared_error as mse
from torchvision import transforms as TT

from src.utils.misc import *

from .modules.data_module_base import DataModuleBase, data_module_register
from .modules.media_rw import load_video

MEAN = {
    'train': [32.736, 32.852, 33.105],
    'val': [33.011, 33.134, 33.425],  # not used
    'test': [33.006, 33.112, 33.371],  # not used
}
MEAN_GRAY = {  # (0.2989 * r + 0.587 * g + 0.114 * b)
    'train': [32.843] * 3,
    'val': [33.127] * 3,  # not used
    'test': [33.107] * 3,  # not used
}
STD = {
    'train': [49.942, 50.019, 50.271],
    'val': [50.235, 50.316, 50.592],  # not used
    'test': [50.455, 50.527, 50.752],  # not used
}
STD_GRAY = {
    'train': [50.020] * 3,
    'val': [50.318] * 3,  # not used
    'test': [50.526] * 3,  # not used
}
EF_MEAN = {
    'dummy': 0.000,
    'train': 55.777,
    'val': 55.825,  # not used
    'test': 55.494,  # not used
}

EF_STD = {
    'dummy': 1.000,
    'train': 12.409,
    'val': 12.305,  # not used
    'test': 12.229,  # not used
}

EF_MIN = {
    'train': 6.907,
    'val': 9.482,  # not used
    'test': 10.193,  # not used
}

EF_MAX = {
    'train': 96.967,
    'val': 81.415,  # not used
    'test': 84.452,  # not used
}

__choice = 'dummy'
_EF_MEAN = EF_MEAN[__choice]
_EF_STD = EF_STD[__choice]
_EF_MEAN_OFFSET = EF_MEAN['train'] - _EF_MEAN
assert _EF_MEAN == 0.000 and _EF_STD == 1.000

def _video_collate_fn(x):
    """Collate function for Pytorch dataloader to merge multiple videos.

    This function should be used in a dataloader for a dataset that returns
    a video as the first element, along with some (non-zero) tuple of
    targets. Then, the input x is a list of tuples:
      - x[i][0] is the i-th video in the batch
      - x[i][1] are the targets for the i-th video

    This function returns a 3-tuple:
      - The first element is the videos concatenated along the frames
        dimension. This is done so that videos of different lengths can be
        processed together (tensors cannot be "jagged", so we cannot have
        a dimension for video, and another for frames).
      - The second element is contains the targets with no modification.
      - The third element is a list of the lengths of the videos in frames.
    """
    video, target = zip(*x)  # Extract the videos and targets

    # ``video'' is a tuple of length ``batch_size''
    #   Each element has shape (channels=3, frames, height, width)
    #   height and width are expected to be the same across videos, but
    #   frames can be different.

    # ``target'' is also a tuple of length ``batch_size''
    # Each element is a tuple of the targets for the item.

    i = list(map(lambda t: t.shape[1], video))  # Extract lengths of videos in frames

    # This contatenates the videos along the the frames dimension (basically
    # playing the videos one after another). The frames dimension is then
    # moved to be first.
    # Resulting shape is (total frames, channels=3, height, width)
    video = torch.as_tensor(np.swapaxes(np.concatenate(video, 1), 0, 1))

    # Swap dimensions (approximately a transpose)
    # Before: target[i][j] is the j-th target of element i
    # After:  target[i][j] is the i-th target of element j
    target = zip(*target)

    return video, target, i


class EchoDynamicDataset(torchvision.datasets.VisionDataset):
    """EchoNet-Dynamic Dataset.

    Args:
        root (string): Root directory of dataset (defaults to `echonet.config.DATA_DIR`)
        split (string): One of {``train'', ``val'', ``test'', ``all'', or ``external_test''}
        target_types (string or list, optional): Type of target to use,
            ``Filename'', ``EF'', ``EDV'', ``ESV'', ``LargeIndex'',
            ``SmallIndex'', ``LargeFrame'', ``SmallFrame'', ``LargeTrace'',
            or ``SmallTrace''
            Can also be a list to output a tuple with all specified target types.
            The targets represent:
                ``Filename'' (string): filename of video
                ``EF'' (float): ejection fraction
                ``EDV'' (float): end-diastolic volume
                ``ESV'' (float): end-systolic volume
                ``LargeIndex'' (int): index of large (diastolic) frame in video
                ``SmallIndex'' (int): index of small (systolic) frame in video
                ``LargeFrame'' (np.array shape=(3, height, width)): normalized large (diastolic) frame
                ``SmallFrame'' (np.array shape=(3, height, width)): normalized small (systolic) frame
                ``LargeTrace'' (np.array shape=(height, width)): left ventricle large (diastolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
                ``SmallTrace'' (np.array shape=(height, width)): left ventricle small (systolic) segmentation
                    value of 0 indicates pixel is outside left ventricle
                             1 indicates pixel is inside left ventricle
            Defaults to ``EF''.
        mean (int, float, or np.array shape=(3,), optional): means for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not shifted).
        std (int, float, or np.array shape=(3,), optional): standard deviation for all (if scalar) or each (if np.array) channel.
            Used for normalizing the video. Defaults to 0 (video is not scaled).
        length (int or None, optional): Number of frames to clip from video. If ``None'', longest possible clip is returned.
            Defaults to 16.
        period (int, optional): Sampling period for taking a clip from the video (i.e. every ``period''-th frame is taken)
            Defaults to 2.
        max_length (int or None, optional): Maximum number of frames to clip from video (main use is for shortening excessively
            long videos when ``length'' is set to None). If ``None'', shortening is not applied to any video.
            Defaults to 250.
        clips (int, optional): Number of clips to sample. Main use is for test-time augmentation with random clips.
            Defaults to 1.
        pad (int or None, optional): Number of pixels to pad all frames on each side (used as augmentation).
            and a window of the original size is taken. If ``None'', no padding occurs.
            Defaults to ``None''.
        noise (float or None, optional): Fraction of pixels to black out as simulated noise. If ``None'', no simulated noise is added.
            Defaults to ``None''.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        external_test_location (string): Path to videos to use for external testing.
    """

    def __init__(
        self, root, transform=None,
        split="train", target_types=["EF"],
        # mean=0., std=1.,
        length=16,
        # max_length=np.inf,
        period=2,
        max_choice_num=1,
        val_test_clips=1,
        pad=None,
        target_transform=None,
        time_pad=0,
        keep_ratio=1.,
        ):

        super().__init__(root, target_transform=target_transform)

        self.split = split
        self.target_types = target_types
        # self.mean = mean.reshape(3, 1, 1, 1) if isinstance(mean, np.ndarray) else mean
        # self.std = std.reshape(3, 1, 1, 1) if isinstance(std, np.ndarray) else std
        self.length = length
        # self.max_length = max_length
        self.period = period
        self.max_choice_num = max_choice_num
        self.val_test_clips = val_test_clips
        self.pad = pad
        self.target_transform = target_transform

        self.file_name_list = []
        self.annotations_list = []
        
        self.time_pad = time_pad

        # Load video-level labels
        self.train = False
        with open(os.path.join(self.root, 'FileList.csv')) as f:
            data = pandas.read_csv(f)
        if isinstance(self.split, list):
            if 'train' in self.split:
                self.train = True
            data = data[data['Split'].isin([s.upper() for s in self.split])]
        elif isinstance(self.split, str):
            if self.split == 'train':
                self.train = True
            data = data[data['Split'] == self.split.upper()]  # filtered by split
            
        
        # data = data[data['Split'] == 'TRAIN']  # filtered by split
        # if split == "train":
        #     data = data[:int(0.8*(len(data)))]
        # else:
        #     data = data[int(0.8*(len(data))):]

        self.header = data.columns.tolist()
        self.file_name_list = data['FileName'].map(lambda x: x + '.avi').tolist()
        self.annotations_list = data.values.tolist()

        # Verify that all video files exist
        missing = set(self.file_name_list) - set(os.listdir(os.path.join(self.root, "Videos")))
        if len(missing) != 0:
            print("{} videos could not be found in {}:".format(len(missing), os.path.join(self.root, "Videos")))
            for f in sorted(missing):
                print("\t", f)
            raise FileNotFoundError(os.path.join(self.root, "Videos", sorted(missing)[0]))

        # Load trace lines of segmentation
        self.file_segment_frame_indexes = defaultdict(list)
        self.file_segment_traces = defaultdict(lambda: defaultdict(list))

        with open(os.path.join(self.root, "VolumeTracings.csv")) as f:
            header = f.readline().strip().split(",")  # first line is header
            assert header == ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]

            for line in f:
                filename, x1, y1, x2, y2, frame = line.strip().split(',')
                frame = int(frame)
                if frame not in self.file_segment_traces[filename]:
                    self.file_segment_frame_indexes[filename].append(frame)  # new one frame as self.frames = {filename: [frame]}
                self.file_segment_traces[filename][frame].append(np.array([x1, y1, x2, y2], dtype=np.float32))  # self.trace = {filename: {frame: [array[x1, y1, x2, y2], ...]]}}
        for filename in self.file_segment_frame_indexes:
            for frame in self.file_segment_frame_indexes[filename]:
                self.file_segment_traces[filename][frame] = np.array(self.file_segment_traces[filename][frame], dtype=np.float32)

        # remove videos missing segment traces XXX: 5 are removed
        keep_list = list(map(lambda f: len(self.file_segment_frame_indexes[f]) >= 2, self.file_name_list))
        self.file_name_list = [f for (f, keep) in zip(self.file_name_list, keep_list) if keep]
        self.annotations_list = [f for (f, keep) in zip(self.annotations_list, keep_list) if keep]
        
        total_len = len(self.file_name_list)
        assert total_len == len(self.annotations_list)
        
        # remove videos according to drop_rate
        if keep_ratio < 1.0:
            assert keep_ratio > 0.
            self.file_name_list = self.file_name_list[:int(total_len * keep_ratio)]
            self.annotations_list = self.annotations_list[:int(total_len * keep_ratio)]
            print(LoggerMisc.block_wrapper(f'length of {self.split} dataset: {total_len} ({len(keep_list) - total_len} videos without segment traces have been removed.)\nKeep ratio: {keep_ratio} -> the final length: {len(self.file_name_list)}'))
        else:
            print(LoggerMisc.block_wrapper(f'length of {self.split} dataset: {total_len} ({len(keep_list) - total_len} videos without segment traces have been removed.)'))
        
        self.transform = transform
        
        # with open('./echonet_dynamic_test_index.csv', 'w') as f:
        #     f.write('file_index,file_name\n')
        #     for i, filename in enumerate(self.file_name_list):
        #         f.write(f'{i},{filename}\n')
        
    def _get_target(self, video, index):
        # Gather targets
        target = {}
        for t in self.target_types:
            filename = self.file_name_list[index]
            if t == "Filename":
                target[t] = self.file_name_list[index]
            elif t == "LargeIndex":
                # Traces are sorted by cross-sectional area
                # Largest (diastolic) frame is last
                target[t] = np.int32(self.file_segment_frame_indexes[filename][-1])
            elif t == "SmallIndex":
                # Smallest (systolic) frame is first
                target[t] = np.int32(self.file_segment_frame_indexes[filename][0])
            elif t == "LargeFrame":
                target[t] = video[self.file_segment_frame_indexes[filename][-1], :, :, :]
            elif t == "SmallFrame":
                target[t] = video[self.file_segment_frame_indexes[filename][0], :, :, :]
            elif t in ["LargeTrace", "SmallTrace"]:
                if t == "LargeTrace":
                    trace = self.file_segment_traces[filename][self.file_segment_frame_indexes[filename][-1]]
                else:
                    trace = self.file_segment_traces[filename][self.file_segment_frame_indexes[filename][0]]
                x1, y1, x2, y2 = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3]
                x = np.concatenate((x1[1:], np.flip(x2[1:])))
                y = np.concatenate((y1[1:], np.flip(y2[1:])))

                r, c = skimage.draw.polygon(np.rint(y).astype(np.int32), np.rint(x).astype(np.int32), (video.shape[2], video.shape[3]))
                mask = np.zeros((video.shape[2], video.shape[3]), np.float32)
                mask[r, c] = 1
                target[t] = mask
            elif t == "EF":  # "EF"
                if self.split == "CLINICAL_TEST" or self.split == "EXTERNAL_TEST":
                    target[t] = np.float32(0)
                else:
                    ef = np.float32(
                        (self.annotations_list[index][self.header.index(t)] - _EF_MEAN) / _EF_STD
                        )
                    # XXX: ef values have been normalized now (currently no change as _EF_MEAN = 0, _EF_STD = 1)
                    target[t] = ef
            elif t == "x":
                target[t] = video
            elif t == "FPS":
                fps = np.float32(
                    (self.annotations_list[index][self.header.index(t)] - 0.0) / 1.0
                    )
                target[t] = fps
            else:
                raise ValueError(f"Target type {t} not recognized")
        return target
    
    def _get_video_clip(self, video, start_index, end_index, largeTrace=None, smallTrace=None):
        video = video[:, start_index:end_index, :, :]
        C, L, H, W = video.shape
        video = video.unsqueeze(0)  # (1, C, L, H, W)
        # Interpolate over D=L, H, W dimensions; here we keep H and W same
        video = torch.nn.functional.interpolate(
            video, size=(self.length, H, W),
            mode='trilinear', align_corners=True,
        )[0]
        
        if self.pad is not None and self.pad > 0:
            # Add padding of zeros (mean color of videos)
            # Crop of original size is taken out
            # (Used as augmentation)
            c, l, h, w = video.shape
            temp = torch.zeros((c, l, h + 2 * self.pad, w + 2 * self.pad), dtype=video.dtype)
            temp[:, :, self.pad:-self.pad, self.pad:-self.pad] = video
            i, j = torch.randint(0, 2 * self.pad, (2,))
            video = temp[:, :, i:(i + h), j:(j + w)]
            
            if largeTrace is not None:
                hh, ww = largeTrace.shape
                assert hh == h and ww == w
                temp = torch.zeros((hh + 2 * self.pad, ww + 2 * self.pad), dtype=largeTrace.dtype)
                temp[self.pad:-self.pad, self.pad:-self.pad] = largeTrace
                largeTrace = temp[i:(i + h), j:(j + w)]
                
            if smallTrace is not None:
                hh, ww = smallTrace.shape
                assert hh == h and ww == w
                temp = torch.zeros((hh + 2 * self.pad, ww + 2 * self.pad), dtype=smallTrace.dtype)
                temp[self.pad:-self.pad, self.pad:-self.pad] = smallTrace
                smallTrace = temp[i:(i + h), j:(j + w)]
            
        return video, largeTrace, smallTrace

    def __getitem__(self, index):
        video_file_name = self.file_name_list[index]
        video_filepath = os.path.join(self.root, "Videos", video_file_name)
        video = load_video(video_filepath)
        
        target = self._get_target(video, index)
        
        video = torch.as_tensor(video, dtype=torch.float).transpose(0, 1) / 255.   # [L, C, HH, WW]
        if self.transform is not None:
            video = self.transform(video).transpose(0, 1)  # [C, L, H, W]
        
        start_index = min(target.get('SmallIndex'), target.get('LargeIndex'))
        end_index = max(target.get('SmallIndex'), target.get('LargeIndex'))
        
        video_clip, largeTrace, smallTrace = self._get_video_clip(
            video,
            start_index,
            end_index,
            torch.as_tensor(target.get('LargeTrace')),
            torch.as_tensor(target.get('SmallTrace')),
            )
        
        if start_index == target.get('SmallIndex'):
            small_index = 0
            large_index = -1
        else:
            small_index = -1
            large_index = 0
        
        return {
            'inputs': {
                'x': torch.as_tensor(video_clip),
                # 'file_name': video_file_name.split('.')[0],  # no file suffix
            },
            'targets': {
                **{'gt_' + k.lower(): torch.as_tensor(v) for k, v in target.items()},
                'file_name': video_file_name.split('.')[0],  # no file suffix
                'index': torch.tensor(index),
                # 'video_frame_real_mask': torch.as_tensor(video_frame_real_mask),
                'small_index': torch.as_tensor(small_index),
                'large_index': torch.as_tensor(large_index),
            }
        }

    def __len__(self):
        return len(self.file_name_list)

    def extra_repr(self) -> str:
        """Additional information to add at end of __repr__."""
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
    

def choose_frames_by_maximizing_mse(video, length, group_length, remainder_for_start, max_choice_num, start):
    def sum_mse_with_list(video, frame_idx_now, frame_idx_list):
        sum_mse = 0
        for frame_idx in frame_idx_list:
            sum_mse += mse(video[:, frame_idx_now], video[:, frame_idx])
        return sum_mse

    frame_idx_range = remainder_for_start + group_length * np.arange(length)
    frame_idx_list = [start]

    for i in range (1, length):
        mse_list = []
        for j in range(frame_idx_range[i - 1], frame_idx_range[i]):
            # sum mse between frame[j] and frames in frames_list
            sum_mse = sum_mse_with_list(video, j, frame_idx_list)
            mse_list.append(sum_mse)
            
        # find the frame index with the maximum mse
        choice_num = min(max_choice_num, len(mse_list))
        max_mse_idx = np.array(mse_list).argsort()[-choice_num:][np.random.randint(choice_num)]
        frame_idx_list.append(frame_idx_range[i - 1] + max_mse_idx)

    return np.array(frame_idx_list)

    
def get_mean_and_std(
    root,
    split='train',
    samples=None,
    num_workers=4,
    ):
    
    dataset = EchoDynamicDataset(
        root=root,
        split=split,
        target_types=['EF'],
        length=None,
        period=1,
        max_length=np.inf,
        # clips=1,
        pad=None,
        noise=None,
        target_transform=None,
    )
    
    if samples is not None and len(dataset) > samples:
        indices = np.random.choice(len(dataset), samples, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices)     
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=num_workers, collate_fn=DataModuleBase.collate_fn)

    n_x = 0  # number of elements taken
    n_ef = 0
    s1 = 0.  # sum of elements along channels (ends up as np.array of dimension (channels,))
    s2 = 0.  # sum of squares of elements along channels (ends up as np.array of dimension (channels,))
    sef1 = 0.
    sef2 = 0.
    max_ef = -np.inf
    min_ef = np.inf
    from tqdm import tqdm
    for batch in tqdm(dataloader):  # x [batch_size, 3, length, H, W]
        x = batch['inputs']['x']
        ef = batch['targets']['gt_ef']
        x = x.transpose(0, 1).reshape(3, -1)  # x [3, batch_size*length*H*W]
        n_x += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x ** 2, dim=1).numpy()
        n_ef += ef.shape[0]
        sef1 += torch.sum(ef).numpy()
        sef2 += torch.sum(ef ** 2).numpy()
        if ef[0] < min_ef:
            min_ef = ef[0]
        if ef[0] > max_ef:
            max_ef = ef[0]
    mean_x = s1 / n_x  # type: np.ndarray
    std_x = np.sqrt(s2 / n_x - mean_x ** 2)  # type: np.ndarray
    mean_ef = sef1 / n_ef  # type: np.ndarray
    std_ef = np.sqrt(sef2 / n_ef - mean_ef ** 2)  # type: np.ndarray

    mean_x = mean_x.astype(np.float32)
    std_x = std_x.astype(np.float32)
    mean_ef = mean_ef.astype(np.float32)
    std_ef = std_ef.astype(np.float32)

    return mean_x, std_x, mean_ef, std_ef, min_ef, max_ef


class MedianGaussianBlur:
    def __init__(self, median_enabled=True, gaussian_enabled=True, kernel_size=3, gaussian_sigma=None):
        self.median_kernel_size = kernel_size
        
        self.gaussian_kernel_size = kernel_size
        self.gaussian_sigma = gaussian_sigma  # 0.8 for kernel_size=3, 1.1 for kernel_size=5, 1.4 for kernel_size=7 by default
        
        self.median_enabled = median_enabled
        self.gaussian_enabled = gaussian_enabled
        
    def _apply_blurs(self, img):
        if self.median_enabled:
            img = cv2.medianBlur(img, self.median_kernel_size)
        if self.gaussian_enabled:
            img = cv2.GaussianBlur(img, (self.gaussian_kernel_size, self.gaussian_kernel_size), self.gaussian_sigma)
        if img.ndim == 2:
            img = img[..., None]
        return img

    def __call__(self, imgs):
        assert isinstance(imgs, torch.Tensor)
        imgs = np.asarray(imgs.permute(0, 2, 3, 1))
        imgs = [self._apply_blurs(img) for img in imgs]
        imgs = np.stack(imgs, axis=0)
        imgs = torch.as_tensor(imgs).permute(0, 3, 1, 2)
        return imgs
    

@data_module_register('echo_dynamic_seg')
class EchoDynamicDataModule(DataModuleBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # XXX: use train mean and std for val and test
        self.data_dir = cfg.data.data_dir
        self.kwargs = {
            'target_types': cfg.data.target_types,
            # 'mean': np.array(cfg.data.mean if cfg.data.mean is not None else MEAN['train']),
            # 'std': np.array(cfg.data.std if cfg.data.std is not None else STD['train']),
            'length': cfg.data.length,
            'max_choice_num': cfg.data.max_choice_num
            }
        self.train_period = cfg.data.period
        self.val_period = getattr(cfg.data, 'val_period', self.train_period)
        self.train_pad = getattr(cfg.data, 'train_pad', 12)
        self.train_time_pad = getattr(cfg.data, 'train_time_pad', 0)
        self.trainset_keep_ratio = getattr(cfg.data, 'trainset_keep_ratio', 1.)
        self.special_train_split = getattr(cfg.data, 'special_train_split', None)
        
        if getattr(cfg.data, 'trainset_augmentation', False):
            warnings.warn('Trainset augmentation is enabled, which will make segmentation labels incorrect.')
            self.train_transform = TT.Compose([
                TT.Grayscale(num_output_channels=1),
                MedianGaussianBlur(
                    median_enabled=getattr(cfg.data, 'median_blur', False),
                    gaussian_enabled=getattr(cfg.data, 'gaussian_blur', False),
                    kernel_size=3,
                    gaussian_sigma=getattr(cfg.data, 'gaussian_sigma', 0.6),
                    ),
                TT.RandomAffine(degrees=(-15, 15), translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-10, 10, -10, 10)),
                TT.Resize((cfg.data.resize_to, cfg.data.resize_to), antialias=True),
                # TT.RandomHorizontalFlip(p=0.5),
                # TT.RandomVerticalFlip(p=0.5),
                TT.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.0, hue=0.0),
                TT.Normalize([i / 255. for i in MEAN_GRAY['train'][:1]], [i / 255. for i in STD_GRAY['train'][:1]], inplace=True),
            ])
        else:
            self.train_transform = TT.Compose([
                TT.Grayscale(num_output_channels=1),
                MedianGaussianBlur(
                    median_enabled=getattr(cfg.data, 'median_blur', False),
                    gaussian_enabled=getattr(cfg.data, 'gaussian_blur', False),
                    kernel_size=3,
                    gaussian_sigma=getattr(cfg.data, 'gaussian_sigma', 0.6),
                    ),
                TT.Resize((cfg.data.resize_to, cfg.data.resize_to), antialias=True),
                TT.Normalize([i / 255. for i in MEAN_GRAY['train'][:1]], [i / 255. for i in STD_GRAY['train'][:1]], inplace=True),
            ])
        
        self.val_transform = TT.Compose([
            TT.Grayscale(num_output_channels=1),
            MedianGaussianBlur(
                median_enabled=getattr(cfg.data, 'median_blur', False),
                gaussian_enabled=getattr(cfg.data, 'gaussian_blur', False),
                kernel_size=3,
                gaussian_sigma=getattr(cfg.data, 'gaussian_sigma', 0.6),
                ),
            TT.Resize((cfg.data.resize_to, cfg.data.resize_to), antialias=True),
            TT.Normalize([i / 255. for i in MEAN_GRAY['train'][:1]], [i / 255. for i in STD_GRAY['train'][:1]], inplace=True),
        ])

    def build_train_dataset(self):
        split = self.special_train_split if self.special_train_split is not None else "train"
        return EchoDynamicDataset(root=self.data_dir, split=split, **self.kwargs,
                                  pad=self.train_pad,
                                  time_pad=self.train_time_pad,
                                  transform=self.train_transform,
                                  period=self.train_period,
                                  keep_ratio=self.trainset_keep_ratio,
                                  )  # XXX: padding for trainset -> data augmentation.
        
    def build_val_dataset(self):
        return EchoDynamicDataset(root=self.data_dir, split="val", **self.kwargs,
                                  transform=self.val_transform,
                                  period=self.val_period,
                                  val_test_clips=getattr(self.cfg.data, 'val_test_clips', 1),
                                  )
        
    def build_test_dataset(self):            
        return EchoDynamicDataset(root=self.data_dir, split="test", **self.kwargs,
                                  transform=self.val_transform,
                                  period=self.val_period,
                                  val_test_clips=getattr(self.cfg.data, 'val_test_clips', 1),
                                  )
    