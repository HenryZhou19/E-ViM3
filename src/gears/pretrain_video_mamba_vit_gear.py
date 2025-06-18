import os
import time
from glob import glob

import numpy as np
import torch

from src.utils.misc import *

from .modules.trainer_base import TrainerBase, trainer_register


@trainer_register('pretrain_video_mamba_vit')
class Trainer(TrainerBase):
    def _before_all_epochs(self, **kwargs):
        super()._before_all_epochs(**kwargs)
        
        assert self.cfg.trainer.dist_eval, 'dist_eval should be True for validation for this pretraining gear.'
        self.min_hold_memory_mb = self.cfg.trainer.min_hold_memory_mb
        self.new_cycle = True
        
    def _before_one_epoch(self, **kwargs):
        # self.epoch == 0 here before the first epoch
        super()._before_one_epoch(**kwargs)
        # self.epoch == self.epoch + 1 after the "super.()..."

        if self.new_cycle:
            if hasattr(self, 'memory_tensor'):
                del self.memory_tensor
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
    def _after_first_train_iter(self, **kwargs):
        super()._after_first_train_iter(**kwargs)
        
        if self.new_cycle and DistMisc.is_dist_avail_and_initialized():
            _, max_allocated_mb, reserved_mb, _ = TensorMisc.get_gpu_memory_usage(verbose=False)
            print(LoggerMisc.block_wrapper(f'Epoch {self.epoch}\n\tMax allocated memory: {max_allocated_mb:.2f} MB\n\tReserved memory: {reserved_mb:.2f} MB\n'))
            if reserved_mb < self.min_hold_memory_mb:
                self.memory_tensor = TensorMisc.allocate_memory_to_tensor(self.min_hold_memory_mb - reserved_mb)
            self.new_cycle = False
            
    def _mixup_data(self, inputs, targets, alpha=1.0):
        x = inputs['x']
        if targets.get('gt_ef', None) is not None:
            label = 'gt_ef'
        elif targets.get('y', None) is not None:
            label = 'y'
        else:
            raise NotImplementedError
        y = targets[label]
        
        batch_size = x.shape[0]
        
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha, size=(batch_size,))
            lam = torch.as_tensor(lam, dtype=x.dtype, device=x.device)
        else:
            lam = torch.ones(batch_size, dtype=x.dtype, device=x.device)
        
        index = torch.randperm(batch_size, device=x.device)
        
        lam = lam.view(-1, *[1]*(x.dim() - 1))
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        lam = lam.view(-1, *[1]*(y.dim() - 1))
        mixed_y = lam * y + (1 - lam) * y[index]
        
        inputs['x'] = mixed_x
        targets[label] = mixed_y
        return inputs, targets


    def _forward(self, batch: dict):
        time.sleep(self.breath_time)
        
        batch: dict = TensorMisc.to(batch, self.device, non_blocking=self.cfg.env.pin_memory)
        inputs: dict = batch['inputs']
        targets: dict = batch['targets']
        
        if self.training:
            if getattr(self.cfg.data, 'mixup', False):
                inputs, targets = self._mixup_data(inputs, targets)
            
            with self.train_autocast():
                outputs = self.model(inputs)
                loss, metrics_dict = self.criterion(outputs, targets)
        else:
            with torch.no_grad():
                with self.val_autocast():
                    outputs = self.model(inputs)
                    loss, metrics_dict = self.criterion(outputs, targets)
                    
                    if self.ema_container is not None:
                        ema_outputs = self.ema_container(inputs)
                        _, ema_metrics_dict = self.ema_criterion(ema_outputs, targets)
                        outputs.update(LoggerMisc.set_dict_key_prefix(ema_outputs, 'ema_'))
                        metrics_dict.update(ema_metrics_dict)
            
        return outputs, loss, metrics_dict
    
    def _after_validation(self, **kwargs):
        LoggerMisc.logging(self.loggers, 'val_epoch', self.last_val_metrics, self.trained_iters)
        
        if 'loss' in self.best_val_metrics:
            if self.last_val_metrics['loss'] >= self.cfg.trainer.loss_spike_multiplier * self.best_val_metrics['loss']:
                print(LoggerMisc.block_wrapper(
                    f'Epoch {self.epoch}, loss spike detected.\n\tnow: {self.last_val_metrics["loss"]:.4f}\n\tbest: {self.best_val_metrics["loss"]:.4f}'
                    ))
                self._load_best_checkpoint_with_new_optimizer()
            else:
                self._save_best_full_checkpoint()
        else:
            self._save_best_full_checkpoint()
            
    def _load_best_checkpoint(self):
        DistMisc.barrier()
        checkpoint_path = glob(os.path.join(self.cfg.info.work_dir, 'checkpoint_best_epoch_*.pth'))
        print(LoggerMisc.block_wrapper(f'Rank {DistMisc.get_rank()}: loading the checkpoint from {checkpoint_path}', '>'), force=True)
        assert len(checkpoint_path) == 1, f'Found {len(checkpoint_path)} checkpoints, please check.'
        checkpoint = torch.load(checkpoint_path[0], map_location='cpu')
        self.model_without_ddp.load_state_dict(checkpoint['model'])
        if self.ema_container is not None:
            assert 'ema_container' in checkpoint or 'ema_model' in checkpoint, 'checkpoint does not contain "ema_container" or "ema_model".'
            if 'ema_container' in checkpoint:
                self.ema_container.load_state_dict(checkpoint['ema_container'])
            else:  # FIXME: deprecated
                self.ema_container.load_state_dict(checkpoint['ema_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if self.scaler is not None:
            assert 'scaler' in checkpoint, 'checkpoint does not contain "scaler".'
            self.scaler.load_state_dict(checkpoint['scaler'])
        # self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_metrics = checkpoint.get('best_val_metrics', {})
        self.last_val_metrics = checkpoint.get('last_val_metrics', {})
        # self.trained_iters = checkpoint['epoch'] * self.train_len
        # self.epoch = self.start_epoch - 1  # will be the same as {checkpoint['epoch'] + 1} by doing '+1' in "before_one_epoch"
        
    def _load_best_checkpoint_with_new_optimizer(self):
        DistMisc.barrier()
        checkpoint_path = glob(os.path.join(self.cfg.info.work_dir, 'checkpoint_best_epoch_*.pth'))
        print(LoggerMisc.block_wrapper(f'Rank {DistMisc.get_rank()}: loading the checkpoint from {checkpoint_path}', '>'), force=True)
        assert len(checkpoint_path) == 1, f'Found {len(checkpoint_path)} checkpoints, please check.'
        checkpoint = torch.load(checkpoint_path[0], map_location='cpu')
        self.model_without_ddp.load_state_dict(checkpoint['model'])
        if self.ema_container is not None:
            assert 'ema_container' in checkpoint or 'ema_model' in checkpoint, 'checkpoint does not contain "ema_container" or "ema_model".'
            if 'ema_container' in checkpoint:
                self.ema_container.load_state_dict(checkpoint['ema_container'])
            else:  # FIXME: deprecated
                self.ema_container.load_state_dict(checkpoint['ema_model'])
        import collections
        self.optimizer.state = collections.defaultdict(dict)
        # self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if self.scaler is not None:
            assert 'scaler' in checkpoint, 'checkpoint does not contain "scaler".'
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        # self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_metrics = checkpoint.get('best_val_metrics', {})
        self.last_val_metrics = checkpoint.get('last_val_metrics', {})
        # self.trained_iters = checkpoint['epoch'] * self.train_len
        # self.epoch = self.start_epoch - 1  # will be the same as {checkpoint['epoch'] + 1} by doing '+1' in "before_one_epoch"
        
    def _save_best_full_checkpoint(self):
        # called in "after_validation"        
        self.best_val_metrics, last_is_best = self.criterion.choose_best(
            self.last_val_metrics, self.best_val_metrics
            )
        
        if DistMisc.is_main_process():
            epoch_finished = self.epoch
            
            if last_is_best:
                save_dict = {
                    'model': self.model_without_ddp.state_dict(),
                    'ema_container': self.ema_container.state_dict() if self.ema_container is not None else None,
                    'scaler': self.scaler.state_dict() if self.scaler is not None else None,
                    'optimizer': self.optimizer.state_dict(),
                    'lr_scheduler': self.lr_scheduler.state_dict(),
                    'best_val_metrics': self.best_val_metrics,
                    'epoch': epoch_finished,
                }
                self._save_or_update_checkpoint(save_dict, self.cfg.info.work_dir, epoch_finished, 'best')
    