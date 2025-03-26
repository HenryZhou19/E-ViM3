import numpy as np
import torch
from torch import nn

from .modules.criterion_base import CriterionBase, criterion_register


@criterion_register('buv')
class BuvCriterion(CriterionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        if getattr(cfg.criterion, 'weighted_bce_loss', 0.) > 0:
            pos_weight = torch.tensor([58/88], dtype=torch.float)
            self.ce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.ce_loss = nn.BCEWithLogitsLoss()
        self.epoch_gt = []
        self.epoch_pred = []
        
        self.none_final_aggregate = getattr(cfg.model, 'final_aggregate_mode', 'all_mean') == 'none'
        self.mixup_train = getattr(cfg.data, 'mixup', False)

    def _get_iter_loss_and_metrics(self, outputs, targets):
        pred = outputs['pred_y']  # [N, 1] or [N, n_global, 1]
        gt = targets['y']  # [N]
        if self.none_final_aggregate:
            gt = gt.unsqueeze(1).unsqueeze(1).expand_as(pred)  # [N, n_global, 1]
        else:
            gt = gt.unsqueeze(1).expand_as(pred)  # [N, 1]
        
        # metrics (loss) used for backprop
        ce_loss = self.ce_loss(pred, gt)
        
        if self.loss_config == 'ce':
            loss = 1 * ce_loss
        else:
            raise NotImplementedError(f'loss "{self.loss_config}" has not been implemented yet.')
        
        if self.none_final_aggregate:
            gt = gt.mean(dim=1)
            pred = pred.mean(dim=1)

        self.epoch_gt.append(gt.detach())
        self.epoch_pred.append(pred.detach())
        
        return loss, {
            'ce_loss': ce_loss,
            }


    def _get_epoch_metrics_and_reset(self):
        if self.mixup_train and self.training:
            self.epoch_gt = []
            self.epoch_pred = []
            return {}
        
        from src.utils.misc import DistMisc
        
        self.epoch_gt = torch.cat(self.epoch_gt, dim=0)
        self.epoch_pred = torch.cat(self.epoch_pred, dim=0)
        
        if self._if_gather_epoch_metrics():
            self.epoch_gt = DistMisc.all_gather(self.epoch_gt, concat_out=True)
            self.epoch_pred = DistMisc.all_gather(self.epoch_pred, concat_out=True)

        return_dict = eval_more(self.epoch_pred, self.epoch_gt)
        
        self.epoch_gt = []
        self.epoch_pred = []
            
        return return_dict
    

@criterion_register('buv_sfusion')
class BuvSimpleFusionCriterion(BuvCriterion):
    pass


@criterion_register('buv_with_frame')
class BuvWithFrameCriterion(CriterionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        if getattr(cfg.criterion, 'weighted_bce_loss', 0.) > 0:
            pos_weight = torch.tensor([58/88], dtype=torch.float)
            self.ce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.ce_loss = nn.BCEWithLogitsLoss()
        self.epoch_gt = []
        self.epoch_pred = []
        self.epoch_f_mean_pred = []
        self.frame_loss_weight = getattr(cfg.criterion, 'frame_loss_weight', 1.0)
        
        self.none_final_aggregate = getattr(cfg.model, 'final_aggregate_mode', 'all_mean') == 'none'
        self.mixup_train = getattr(cfg.data, 'mixup', False)


    def _get_iter_loss_and_metrics(self, outputs, targets):
        pred = outputs['pred_y']  # [N, 1] or [N, n_global, 1]
        pred_frame = outputs['pred_frame_y']  # [N, L, 1] or [N, L, n_global_frame, 1]
        gt = targets['y']  # [N]
        if self.none_final_aggregate:
            gt_frame = gt.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(pred_frame)  # [N, L, n_global_frame, 1]
            gt = gt.unsqueeze(1).unsqueeze(1).expand_as(pred)  # [N, n_global, 1]
        else:
            gt_frame = gt.unsqueeze(1).unsqueeze(1).expand_as(pred_frame)  # [N, L, 1]
            gt = gt.unsqueeze(1).expand_as(pred)  # [N, 1]
        
        # metrics (loss) used for backprop
        ce_loss = self.ce_loss(pred, gt)
        frame_ce_loss = self.ce_loss(pred_frame, gt_frame)
        
        if self.loss_config == 'ce':
            loss = 1 * ce_loss + self.frame_loss_weight * frame_ce_loss
        else:
            raise NotImplementedError(f'loss "{self.loss_config}" has not been implemented yet.')
        
        if self.none_final_aggregate:
            gt = gt.mean(dim=1)
            pred = pred.mean(dim=1)
            pred_frame = pred_frame.mean(dim=2)

        self.epoch_gt.append(gt.detach())
        self.epoch_pred.append(pred.detach())
        self.epoch_f_mean_pred.append(pred_frame.mean(dim=1).detach())
        
        return loss, {
            'ce_loss': ce_loss,
            'frame_ce_loss': frame_ce_loss,
            }


    def _get_epoch_metrics_and_reset(self):
        if self.mixup_train and self.training:
            self.epoch_gt = []
            self.epoch_pred = []
            self.epoch_f_mean_pred = []
            return {}
        
        from src.utils.misc import DistMisc
        
        self.epoch_gt = torch.cat(self.epoch_gt, dim=0)
        self.epoch_pred = torch.cat(self.epoch_pred, dim=0)
        self.epoch_f_mean_pred = torch.cat(self.epoch_f_mean_pred, dim=0)
        
        if self._if_gather_epoch_metrics():
            
            self.epoch_gt = DistMisc.all_gather(self.epoch_gt, concat_out=True)
            self.epoch_pred = DistMisc.all_gather(self.epoch_pred, concat_out=True)
            self.epoch_f_mean_pred = DistMisc.all_gather(self.epoch_f_mean_pred, concat_out=True)

        return_dict = eval_more(self.epoch_pred, self.epoch_gt)
        return_dict.update(eval_more(self.epoch_f_mean_pred, self.epoch_gt, 'frame_'))
        
        self.epoch_gt = []
        self.epoch_pred = []
        self.epoch_f_mean_pred = []
            
        return return_dict
        
        
def eval_more(output, target, prefix=''):
       
    auc, opt_accuracy, opt_recall, opt_specificity, opt_f1, opt_precision, opt_threshold = precision_recall_f1(output, target)
    
    return {
        f'{prefix}auc': auc,
        f'{prefix}opt_threshold': opt_threshold,
        f'{prefix}opt_F1': opt_f1,
        f'{prefix}opt_recall': opt_recall,
        f'{prefix}opt_specificity': opt_specificity,
        f'{prefix}opt_precision': opt_precision,
        f'{prefix}opt_accuracy': opt_accuracy,
    }
        
@torch.no_grad()
def precision_recall_f1(pred: torch.Tensor, target: torch.Tensor): # from mmclassification
    from sklearn.metrics import roc_auc_score, roc_curve
    
    y_pred = torch.sigmoid(pred.float()).cpu().numpy().flatten()
    y_true = target.cpu().numpy().flatten()  # 0 or 1
    auc = roc_auc_score(y_true, y_pred)
    
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    J = tpr - fpr
    optimal_idx = np.argmax(J)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_lable = y_pred > optimal_threshold - 1e-7
    TP = ((y_pred_lable == 1) * (y_true == 1)).sum()
    FP = ((y_pred_lable == 1) * (y_true == 0)).sum()
    TN = ((y_pred_lable == 0) * (y_true == 0)).sum()
    FN = ((y_pred_lable == 0) * (y_true == 1)).sum()
    
    opt_accuracy = 100. * (TP + TN) / len(y_true)
    opt_precision = 100. * TP / ((TP + FP) + ((TP + FP) == 0) * 1)
    opt_recall = 100. * TP / (TP + FN)
    opt_specificity = 100. * TN / (TN + FP)
    if opt_precision == 0 or opt_recall == 0:
        opt_f1 = 0
    else:
        opt_f1 = 2 / (1 / opt_precision + 1 / opt_recall)

    return float(auc), float(opt_accuracy), float(opt_recall), float(opt_specificity), float(opt_f1), float(opt_precision), float(optimal_threshold)