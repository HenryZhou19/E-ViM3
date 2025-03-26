import os

import torch
from torch import nn

from src.datasets.echo_dynamic_dataset import _EF_STD

from .modules.criterion_base import CriterionBase, criterion_register
from .modules.losses import *


class EchoDynamicCriterionBase(nn.Module):
    def __init__(self, loss_choice, softloss_tolerance=0.0, smooth_l1_beta=1.0, file_save_path=None):
        super().__init__()
        
        self.loss_choice = loss_choice
        self.softloss_tolerance = softloss_tolerance
        if self.softloss_tolerance > 0.0:
            self.soft_loss = True
            self.mse_loss = SoftMSELoss(abs_tolerance=self.softloss_tolerance)
            self.l1_loss = SoftL1Loss(abs_tolerance=self.softloss_tolerance)
            self.smooth_l1_loss = SoftSmoothL1Loss(abs_tolerance=self.softloss_tolerance, beta=smooth_l1_beta)
            
            self.real_mse_loss = nn.MSELoss()
            self.real_l1_loss = nn.L1Loss()
            # self.real_smooth_l1_loss = nn.SmoothL1Loss(beta=smooth_l1_beta)
        else:
            self.soft_loss = False
            self.mse_loss = nn.MSELoss()
            self.l1_loss = nn.L1Loss()
            self.smooth_l1_loss = nn.SmoothL1Loss(beta=smooth_l1_beta)
            
        self.gt_ef_all = []
        self.pred_ef_all = []
        self.file_index_all = []
        self.file_save_path = file_save_path

    def forward(self, pred_ef, gt_ef, file_index=None):
        # gt_ef = targets['gt_ef']  # [N]
        # pred_ef = outputs['pred_ef']  # [N, 1 or n_samples]
        assert gt_ef.dim() == 1
        assert pred_ef.dim() == 2
        if not self.training:
            self.gt_ef_all.append(gt_ef)
            self.pred_ef_all.append(pred_ef.mean(1))
            file_index = file_index if file_index is not None else -torch.ones(gt_ef)
            self.file_index_all.append(file_index)
        
        # metrics (loss) used for backprop
        gt_ef_repeat = gt_ef.unsqueeze(1).repeat(1, pred_ef.shape[1])  # [N, n_samples]
        mse_loss = self.mse_loss(pred_ef, gt_ef_repeat)
        l1_loss = self.l1_loss(pred_ef, gt_ef_repeat)
        smooth_l1_loss = self.smooth_l1_loss(pred_ef, gt_ef_repeat)
        
        if self.loss_choice == 'mse':
            loss = 1 * mse_loss
        elif self.loss_choice == 'l1':
            loss = 1 * l1_loss
        elif self.loss_choice == 'smooth_l1':
            loss = 1 * smooth_l1_loss
        else:
            raise NotImplementedError(f'loss "{self.loss_choice}" has not been implemented yet.')
        
        if self.soft_loss:
            with torch.no_grad():
                mse_loss_show = self.real_mse_loss(pred_ef, gt_ef_repeat)
                l1_loss_show = self.real_l1_loss(pred_ef, gt_ef_repeat)
                # smooth_l1_loss_show = self.real_smooth_l1_loss(pred_ef, gt_ef_repeat)
        else:
            mse_loss_show = mse_loss
            l1_loss_show = l1_loss
            # smooth_l1_loss_show = smooth_l1_loss
        
        return loss, {
            'mse_loss': mse_loss_show * (_EF_STD ** 2),
            'L1_loss': l1_loss_show * _EF_STD,
            }
        
    def _get_epoch_metrics_and_reset(self, infer_mode, ema_mode, _if_gather_epoch_metrics):
        if self.training:
            return_dict = {}
        else:
            import numpy as np
            import sklearn.metrics

            from src.utils.misc import DistMisc
            
            self.gt_ef_all = torch.cat(self.gt_ef_all, dim=0)
            self.pred_ef_all = torch.cat(self.pred_ef_all, dim=0)
            
            if _if_gather_epoch_metrics:
                self.gt_ef_all = DistMisc.all_gather(self.gt_ef_all, concat_out=True)
                self.pred_ef_all = DistMisc.all_gather(self.pred_ef_all, concat_out=True)
            
            if infer_mode:
                self.file_index_all = torch.cat(self.file_index_all, dim=0)

                if _if_gather_epoch_metrics:
                    self.file_index_all = DistMisc.all_gather(self.file_index_all, concat_out=True)
                    
                if DistMisc.is_main_process():
                    if self.file_save_path is not None:
                        ema_str = 'ema_' if ema_mode else ''
                        with open(os.path.join(self.file_save_path, ema_str + 'EF_results.csv'), 'w') as f:
                            f.write('file_index,gt_ef,pred_ef\n')
                            for i in range(len(self.gt_ef_all)):
                                f.write(f'{self.file_index_all[i].item()},{self.gt_ef_all[i].item()},{self.pred_ef_all[i].item()}\n')
            
            self.gt_ef_all = (self.gt_ef_all * _EF_STD).float().cpu().numpy()
            self.pred_ef_all = (self.pred_ef_all * _EF_STD).float().cpu().numpy()

            skl_corr = np.corrcoef(self.gt_ef_all, self.pred_ef_all)[0, 1]
            skl_R2 = sklearn.metrics.r2_score(self.gt_ef_all, self.pred_ef_all)
            skl_L1 = sklearn.metrics.mean_absolute_error(self.gt_ef_all, self.pred_ef_all)
            
            return_dict = {
                'skl_corr': skl_corr,
                'skl_R2': skl_R2,
                'skl_L1': skl_L1,
                }
            
        self.gt_ef_all = []
        self.pred_ef_all = []
        
        return return_dict

@criterion_register('echo_dynamic')
class EchoDynamicCriterion(CriterionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.criterion = EchoDynamicCriterionBase(
            loss_choice=self.loss_config,  # from CriterionBase
            softloss_tolerance=getattr(cfg.criterion, 'softloss_tolerance', 0.0),
            smooth_l1_beta=getattr(cfg.criterion, 'smooth_l1_beta', 1.0),
            file_save_path=cfg.info.work_dir,
            )

    def _get_iter_loss_and_metrics(self, outputs, targets):
        gt_ef = targets['gt_ef']  # [N]
        pred_ef = outputs['pred_ef']  # [N, 1 or n_samples]
        
        loss, return_dict = self.criterion(pred_ef, gt_ef, targets['index'])
        
        return loss, return_dict
    
    def _get_epoch_metrics_and_reset(self):
        return self.criterion._get_epoch_metrics_and_reset(infer_mode=self.infer_mode, ema_mode=self.ema_mode, _if_gather_epoch_metrics=self._if_gather_epoch_metrics())


@criterion_register('echo_dynamic_with_trace')
class EchoDynamicCriterion(CriterionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.criterion = EchoDynamicCriterionBase(
            loss_choice=self.loss_config,  # from CriterionBase
            softloss_tolerance=getattr(cfg.criterion, 'softloss_tolerance', 0.0),
            smooth_l1_beta=getattr(cfg.criterion, 'smooth_l1_beta', 1.0),
            file_save_path=cfg.info.work_dir,
            )
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        
        self.bce_weight = getattr(cfg.criterion, 'bce_weight', 0.5)
        self.dice_weight = getattr(cfg.criterion, 'dice_weight', 0.5)

    def _get_iter_loss_and_metrics(self, outputs, targets):
        gt_ef = targets['gt_ef']  # [N]
        pred_ef = outputs['pred_ef']  # [N, 1 or n_samples]
        
        loss, return_dict = self.criterion(pred_ef, gt_ef, targets['index'])
        
        frame_indices = targets['frame_indices']  # [N, L]
        largeindex = targets['largeindex']  # [N]
        smallindex = targets['smallindex']  # [N]
        largetrace = targets['largetrace']  # [N, H, W]
        smalltrace = targets['smalltrace']  # [N, H, W]
        
        pred_trace = outputs['pred_trace']  # [N, L, H, W]  logits
        
        N = pred_trace.shape[0]
        pred_exist_trace = []
        label_exist_trace = []
        for n in range(N):
            if largeindex[n] in frame_indices[n]:
                pred_largetrace = pred_trace[n, frame_indices[n] == largeindex[n]]  # [1, H, W]
                assert pred_largetrace.shape[0] == 1
                pred_exist_trace.append(pred_largetrace.squeeze(0))  # [H, W]
                label_exist_trace.append(largetrace[n])  # [H, W]
                
            if smallindex[n] in frame_indices[n]:
                pred_smalltrace = pred_trace[n, frame_indices[n] == smallindex[n]]  # [1, H, W]
                assert pred_smalltrace.shape[0] == 1
                pred_exist_trace.append(pred_smalltrace.squeeze(0))  # [H, W]
                label_exist_trace.append(smalltrace[n])  # [H, W]
        
        if len(pred_exist_trace) > 0:
            pred_exist_trace = torch.stack(pred_exist_trace, dim=0)  # [N_exist, H, W]
            label_exist_trace = torch.stack(label_exist_trace, dim=0)
            
            bce_loss = self.bce_loss(pred_exist_trace, label_exist_trace)
            dice_loss = self.dice_loss(torch.sigmoid(pred_exist_trace), label_exist_trace)
        else:
            bce_loss = torch.tensor(0.0, device=pred_trace.device)
            dice_loss = torch.tensor(0.0, device=pred_trace.device)
        
        return_dict.update({
            'bce_loss': bce_loss,
            'dice_loss': dice_loss,
            })
        
        loss = loss + self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return loss, return_dict
    
    def _get_epoch_metrics_and_reset(self):
        return self.criterion._get_epoch_metrics_and_reset(infer_mode=self.infer_mode, ema_mode=self.ema_mode, _if_gather_epoch_metrics=self._if_gather_epoch_metrics())