import os

import numpy as np
import torch
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn

from src.datasets.echo_dynamic_dataset import _EF_STD

from .modules.criterion_base import CriterionBase, criterion_register
from .modules.dice_and_ce_loss import DC_and_CE_loss_Full_Out
from .modules.losses import *

EPS = 1e-8


@criterion_register('echo_dynamic_seg')
class EchoDynamicSegCriterion(CriterionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.batch_dice = False
        print('Using sample dice loss + CE loss')
        self.dc_ce_loss = DC_and_CE_loss_Full_Out(
            {'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False},
            {}
            )
        self._init_epoch_metrics()
                        
    def _init_epoch_metrics(self):
        self.es_epoch_foreground_dice = []
        self.es_epoch_tp = []
        self.es_epoch_fp = []
        self.es_epoch_fn = []
        self.ed_epoch_foreground_dice = []
        self.ed_epoch_tp = []
        self.ed_epoch_fp = []
        self.ed_epoch_fn = []
    
    @torch.no_grad()
    def _update_epoch_metrics(self, pred_es, gt_es, pred_ed, gt_ed):
        device = pred_es.device
        num_classes = pred_es.shape[1]
        output_softmax = softmax_helper(pred_es)
        pred_es = output_softmax.argmax(1)
        gt_es = gt_es[:, 0]
        axes = tuple(range(1, len(gt_es.shape)))
        tp_hard = torch.zeros((gt_es.shape[0], num_classes - 1), device=device)  # [N, fg_classes]
        fp_hard = torch.zeros((gt_es.shape[0], num_classes - 1), device=device)  # [N, fg_classes]
        fn_hard = torch.zeros((gt_es.shape[0], num_classes - 1), device=device)  # [N, fg_classes]
        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor((pred_es == c).float() * (gt_es == c).float(), axes=axes)
            fp_hard[:, c - 1] = sum_tensor((pred_es == c).float() * (gt_es != c).float(), axes=axes)
            fn_hard[:, c - 1] = sum_tensor((pred_es != c).float() * (gt_es == c).float(), axes=axes)

        tp_hard = tp_hard.sum(0, keepdim=True)  # [1, fg_classes]
        fp_hard = fp_hard.sum(0, keepdim=True)  # [1, fg_classes]
        fn_hard = fn_hard.sum(0, keepdim=True)  # [1, fg_classes]

        self.es_epoch_foreground_dice.append((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + EPS))
        self.es_epoch_tp.append(tp_hard)
        self.es_epoch_fp.append(fp_hard)
        self.es_epoch_fn.append(fn_hard)
        
        output_softmax = softmax_helper(pred_ed)
        pred_ed = output_softmax.argmax(1)
        gt_ed = gt_ed[:, 0]
        axes = tuple(range(1, len(gt_ed.shape)))
        tp_hard = torch.zeros((gt_ed.shape[0], num_classes - 1), device=device)  # [N, fg_classes]
        fp_hard = torch.zeros((gt_ed.shape[0], num_classes - 1), device=device)  # [N, fg_classes]
        fn_hard = torch.zeros((gt_ed.shape[0], num_classes - 1), device=device)  # [N, fg_classes]
        for c in range(1, num_classes):
            tp_hard[:, c - 1] = sum_tensor((pred_ed == c).float() * (gt_ed == c).float(), axes=axes)
            fp_hard[:, c - 1] = sum_tensor((pred_ed == c).float() * (gt_ed != c).float(), axes=axes)
            fn_hard[:, c - 1] = sum_tensor((pred_ed != c).float() * (gt_ed == c).float(), axes=axes)

        tp_hard = tp_hard.sum(0, keepdim=True)  # [1, fg_classes]
        fp_hard = fp_hard.sum(0, keepdim=True)  # [1, fg_classes]
        fn_hard = fn_hard.sum(0, keepdim=True)  # [1, fg_classes]

        self.ed_epoch_foreground_dice.append((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + EPS))
        self.ed_epoch_tp.append(tp_hard)
        self.ed_epoch_fp.append(fp_hard)
        self.ed_epoch_fn.append(fn_hard)
        
        
    def _get_iter_loss_and_metrics(self, outputs, targets):
        pred_seg = outputs['pred_seg']  # [N, C=seg_classes, L, H, W]

        gt_es = targets['gt_smalltrace'].unsqueeze(1)  # [N, C=1, H, W] small
        gt_ed = targets['gt_largetrace'].unsqueeze(1)  # [N, C=1, H, W] large
        
        small_index = targets['small_index']  # [N] 0 or -1
        large_index = targets['large_index']  # [N] 0 or -1
        
        pred_es = []
        pred_ed = []
        for idx, (s_idx, l_idx) in enumerate(zip(small_index, large_index)):
            pred_es.append(pred_seg[idx, :, s_idx, ...])  # [1, C=seg_classes, H, W]
            pred_ed.append(pred_seg[idx, :, l_idx, ...])  # [1, C=seg_classes, H, W]
        
        pred_es = torch.stack(pred_es, dim=0)  # [N, C=seg_classes, H, W]
        pred_ed = torch.stack(pred_ed, dim=0)  # [N, C=seg_classes, H, W]
        
        # print(pred_es.shape)
        # print(gt_es.shape)
        # print(pred_ed.shape)
        # print(gt_ed.shape)

        # metrics (loss) used for backprop
        if self.loss_config == 'dc_and_ce':
            es_dc_ce_loss, es_dc_loss, es_ce_loss = self.dc_ce_loss(pred_es, gt_es)
            ed_dc_ce_loss, ed_dc_loss, ed_ce_loss = self.dc_ce_loss(pred_ed, gt_ed)
            loss = 1 * es_dc_ce_loss + 1 * ed_dc_ce_loss
        else:
            raise NotImplementedError(f'loss "{self.loss_config}" has not been implemented yet.')
    
        # metrics not used for backprop
        es_dice = -es_dc_loss.detach()
        ed_dice = -ed_dc_loss.detach()
        self._update_epoch_metrics(pred_es, gt_es, pred_ed, gt_ed)

        # if self.infer_mode:
        #     
        
        return loss, {
            'es_dc_ce_loss': es_dc_ce_loss.detach(),
            'ed_dc_ce_loss': ed_dc_ce_loss.detach(),
            'es_dice': es_dice,
            'ed_dice': ed_dice,
            }
            
    @torch.no_grad()        
    def _get_epoch_metrics_and_reset(self):
        from src.utils.misc import DistMisc

        self.es_epoch_foreground_dice = torch.cat(self.es_epoch_foreground_dice, dim=0)
        self.es_epoch_tp = torch.cat(self.es_epoch_tp, dim=0)  # [ALL_Batch, fg_classes]
        self.es_epoch_fp = torch.cat(self.es_epoch_fp, dim=0)  # [ALL_Batch, fg_classes]
        self.es_epoch_fn = torch.cat(self.es_epoch_fn, dim=0)  # [ALL_Batch, fg_classes]
        self.ed_epoch_foreground_dice = torch.cat(self.ed_epoch_foreground_dice, dim=0)
        self.ed_epoch_tp = torch.cat(self.ed_epoch_tp, dim=0)  # [ALL_Batch, fg_classes]
        self.ed_epoch_fp = torch.cat(self.ed_epoch_fp, dim=0)  # [ALL_Batch, fg_classes]
        self.ed_epoch_fn = torch.cat(self.ed_epoch_fn, dim=0)  # [ALL_Batch, fg_classes]
        
        if self._if_gather_epoch_metrics():
            self.es_epoch_foreground_dice = DistMisc.all_gather(self.es_epoch_foreground_dice, concat_out=True)
            self.es_epoch_tp = DistMisc.all_gather(self.es_epoch_tp, concat_out=True)
            self.es_epoch_fp = DistMisc.all_gather(self.es_epoch_fp, concat_out=True)
            self.es_epoch_fn = DistMisc.all_gather(self.es_epoch_fn, concat_out=True)
            self.ed_epoch_foreground_dice = DistMisc.all_gather(self.ed_epoch_foreground_dice, concat_out=True)
            self.ed_epoch_tp = DistMisc.all_gather(self.ed_epoch_tp, concat_out=True)
            self.ed_epoch_fp = DistMisc.all_gather(self.ed_epoch_fp, concat_out=True)
            self.ed_epoch_fn = DistMisc.all_gather(self.ed_epoch_fn, concat_out=True)

        self.es_epoch_tp = torch.sum(self.es_epoch_tp, dim=0)  # [fg_classes]
        self.es_epoch_fp = torch.sum(self.es_epoch_fp, dim=0)  # [fg_classes]
        self.es_epoch_fn = torch.sum(self.es_epoch_fn, dim=0)  # [fg_classes]
        self.ed_epoch_tp = torch.sum(self.ed_epoch_tp, dim=0)  # [fg_classes]
        self.ed_epoch_fp = torch.sum(self.ed_epoch_fp, dim=0)  # [fg_classes]
        self.ed_epoch_fn = torch.sum(self.ed_epoch_fn, dim=0)  # [fg_classes]

        es_global_dice = 2 * self.es_epoch_tp / (2 * self.es_epoch_tp + self.es_epoch_fp + self.es_epoch_fn + EPS)
        ed_global_dice = 2 * self.ed_epoch_tp / (2 * self.ed_epoch_tp + self.ed_epoch_fp + self.ed_epoch_fn + EPS)
        overall_global_dice = 2 * (self.es_epoch_tp + self.ed_epoch_tp) / (2 * (self.es_epoch_tp + self.ed_epoch_tp) + self.es_epoch_fp + self.ed_epoch_fp + self.es_epoch_fn + self.ed_epoch_fn + EPS)
        
        return_dict ={}
        return_dict['es_global_dice'] = torch.mean(es_global_dice)
        return_dict['es_each_mean_dice'] = torch.mean(self.es_epoch_foreground_dice)    
        return_dict['ed_global_dice'] = torch.mean(ed_global_dice)
        return_dict['ed_each_mean_dice'] = torch.mean(self.ed_epoch_foreground_dice)
        return_dict['overall_global_dice'] = torch.mean(overall_global_dice)

        self.es_epoch_foreground_dice = []
        self.es_epoch_tp = []
        self.es_epoch_fp = []
        self.es_epoch_fn = []
        self.ed_epoch_foreground_dice = []
        self.ed_epoch_tp = []
        self.ed_epoch_fp = []
        self.ed_epoch_fn = []
            
        return return_dict
