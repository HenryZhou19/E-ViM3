import time

import numpy as np
import torch

from src.utils.misc import *

from .modules.trainer_base import TrainerBase, trainer_register


@trainer_register('echo_dynamic_simple_fuser')
class Trainer(TrainerBase):
    def _before_all_epochs(self, **kwargs):
        super()._before_all_epochs(**kwargs)
        
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
        
        inputs['train_progress'] = self.trained_iters / self.total_iters
        
        if self.training:
            if getattr(self.cfg.data, 'mixup', False):
                inputs, targets = self._mixup_data(inputs, targets)
            
            # assert self.ema_container is not None, 'ema_container is None'
            # inputs['ema_model'] = self.ema_container.ema_model
            # inputs['gt_ef'] = targets['gt_ef']
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