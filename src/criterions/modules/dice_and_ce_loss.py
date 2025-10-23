import numpy as np
import torch
import torch.nn.functional as F
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch import nn


class SymmetricContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, feature_1, feature_2, label_case_1, label_case_2):
        ## feature_x: [N_half, D]
        ## label_case_x: [N_half]
        
        dist = 1 - F.cosine_similarity(feature_1, feature_2, dim=1)  # [N_half]
        label = (label_case_1 == label_case_2).float()  # [N_half]
        
        loss = label * (dist ** 2) + (1 - label) * (torch.clamp(1.0 - dist, min=0.0) ** 2)
        return loss.mean()


class DC_and_CE_loss_Full_Out(DC_and_CE_loss):
    # def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
    #              log_dice=False, ignore_label=None):
    #     """
    #     CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
    #     :param soft_dice_kwargs:
    #     :param ce_kwargs:
    #     :param aggregate:
    #     :param square_dice:
    #     :param weight_ce:
    #     :param weight_dice:
    #     """
    #     super().__init__()
    #     if ignore_label is not None:
    #         assert not square_dice, 'not implemented'
    #         ce_kwargs['reduction'] = 'none'
    #     self.log_dice = log_dice
    #     self.weight_dice = weight_dice
    #     self.weight_ce = weight_ce
    #     self.aggregate = aggregate
    #     self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    #     self.ignore_label = ignore_label

    #     if not square_dice:
    #         self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
    #     else:
    #         self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)
            
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        :param net_output: [N, C=seg_classes, D, H, W]
        :param target: [N, C=1, D, H, W]
        :return:
        """
        if self.ignore_label is not None:  # default is False. If self.ignore_label == 2, then the loss is calculated only for class 0 and 1
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label  # mask == True, where the label is not equal to the ignore_label, 
            target[~mask] = 0  # set all ignore_labels to 0 (background)
            mask = mask.float()
        else:
            mask = None  # default

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0  # [N, C=seg_classes, D, H, W], [N, C=1, D, H, W] -> [
        if self.log_dice:  # default is False
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0  # [N, C=seg_classes, D, H, W], [N, D, H, W] -> [
        if self.ignore_label is not None:  # default is False
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result, dc_loss, ce_loss


class MultipleOutputLoss2(nn.Module):
    def __init__(self, net_num_pool_op_kernel_sizes, batch_dice=True, loss=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        """
        super().__init__()
        
        net_numpool = len(net_num_pool_op_kernel_sizes)  # XXX: for UNet's multi decoder layers

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(net_numpool)])

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        mask = np.array([True] + [True if i < net_numpool - 1 else False for i in range(1, net_numpool)])
        weights[~mask] = 0
        weights = weights / weights.sum()
        self.ds_loss_weights = weights
        
        self.batch_dice = batch_dice
        
        if loss is None:
            self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        else:
            raise NotImplementedError

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[i])
        return l


class ClassLossV2(nn.Module):
    def __init__(self, lamda_class=1.0, epoch_thresh=None, smoothing=0.0):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(ClassLossV2, self).__init__()
        self.lambda_class = lamda_class
        self.epoch_thresh = epoch_thresh
        #self.smooth_loss = LabelSmoothingLoss(3, smoothing=smoothing)

    def forward(self, x, y, pred_class, case_labels=None, epoch=None):
        class_label = []
        for i in range(y[0].shape[0]): # for each batch element
            if (y[0][i] == 2).sum() > 100:
                class_label.append(1)
            elif (y[0][i] == 3).sum() > 100:
                class_label.append(2)
            else:
                class_label.append(0)
        class_label = torch.tensor(class_label).long().cuda()
        case_label = torch.tensor(case_labels).long().cuda()
        class_loss = nn.CrossEntropyLoss()(pred_class, case_label)
        #class_loss = self.smooth_loss(pred_class, class_label)
        if self.epoch_thresh is None or epoch is None:
            loss = class_loss * self.lambda_class
        else:
            if epoch > self.epoch_thresh:
                lambda_class = self.lambda_class
            else:
                lambda_class = epoch / self.epoch_thresh * self.lambda_class
            loss = class_loss * lambda_class
        return loss


class ClassLossV3(ClassLossV2):
    pass