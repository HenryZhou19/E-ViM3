from .modules.criterion_base import CriterionBase, criterion_register
from .modules.losses import *


@criterion_register('pretrain_video_mamba_vit')
class EchoDynamicCriterion(CriterionBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.cfg = cfg
        self.infer_save_dir = cfg.info.work_dir
        
        assert self.loss_config == 'mse', f'Unsupported loss config: {self.loss_config}'
        self.mse_loss = nn.MSELoss()

    def _get_iter_loss_and_metrics(self, outputs, targets):
        original_masked = outputs['original_masked']  # [N_masked, patch_channels_all=C*h*w]
        pred_masked = outputs['pred_masked']  # [N_masked, patch_channels_all=C*h*w]
        
        mse_loss = self.mse_loss(pred_masked, original_masked)
        
        loss = 1.0 * mse_loss
        
        if self.infer_mode:
            self.do_test_process(
                outputs,
                targets['file_name'],
                channels=self.cfg.data.in_channels,
                time_patch_size=getattr(self.cfg.model, 'time_patch_size', 1),
                )
        
        return loss, {
            'mse_loss': mse_loss,
            }


    @torch.no_grad()
    def do_test_process(self, outputs, file_name, channels, time_patch_size, save_as_video=False, do_for_ema=False):
        if not do_for_ema and self.ema_mode:
            return  # do nothing
        
        import os

        import numpy as np

        from src.datasets.echo_dynamic_dataset import MEAN_GRAY, STD_GRAY
        
        assert channels == 1, f'Unsupported channels: {channels}'
        
        # original_masked = outputs['original_masked']  # [N_masked, patch_channels_all=C*h*w]
        pred_masked = outputs['pred_masked'].float()  # [N_masked, patch_channels_all=C*h*w]
        patch_original = outputs['patch_original'].float()  # [N, L_all, num_patches_per_frame, patch_channels_all=C*h*w] or [N, LL, HH, WW, C*l*h*w]
        patch_mask = outputs['patch_mask']  # [N, L_all, num_patches_per_frame] or [N, LL, HH, WW], 'True' for masked, 'False' for not masked
        
        mean = MEAN_GRAY['train'][0]
        std = STD_GRAY['train'][0]
        
        MASK_COLOR = torch.tensor([80, 80, 40], device=patch_original.device, dtype=torch.float).reshape(1, -1, 1)  # [1, 3, 1]
        MASK_COLOR = (MASK_COLOR - mean) / std  # 0~255
        
        if len(patch_original.shape) == 4:
            assert len(patch_mask.shape) == 3
            N, L_all, num_patches_per_frame, patch_channels_all = patch_original.shape
            patch_grid_num = int(num_patches_per_frame ** 0.5)
            patch_original = patch_original.reshape(N, L_all, patch_grid_num, patch_grid_num, patch_channels_all)
            patch_mask = patch_mask.reshape(N, L_all, patch_grid_num, patch_grid_num)
        elif len(patch_original.shape) == 5:
            assert len(patch_mask.shape) == 4
        else:
            raise ValueError(f'Unsupported patch_original shape: {patch_original.shape}')
        N, LL, HH, WW, patch_channels_all = patch_original.shape
        patch_size = int((patch_channels_all / channels / time_patch_size) ** 0.5)
        
        macro_patch_grids = (LL, HH, WW)
        patch_size_grids = (time_patch_size, patch_size, patch_size)
        
        patch_pred = patch_original.clone()
        patch_pred[patch_mask] = pred_masked
        
        patch_mask_color = patch_original.clone()
        patch_mask_color = patch_mask_color.reshape(N, *macro_patch_grids, channels, -1)  # [N, LL, HH, WW, C, l*h*w]
        patch_mask_color = patch_mask_color.expand(-1, -1, -1, -1, 3, -1).clone()  # [N, LL, HH, WW, C=3, l*h*w]
        patch_mask_color[patch_mask] = MASK_COLOR  # [NN, C=3, l*h*w]
        
        patch_original = patch_original.reshape(N, *macro_patch_grids, channels, *patch_size_grids)  # [N, LL, HH, WW, C, l, h, w]
        patch_original = patch_original.expand(-1, -1, -1, -1, 3, -1, -1, -1)
        patch_original = patch_original.permute(0, 1, 5, 4, 2, 6, 3, 7)  # [N, LL, time_patch_size, C, h, patch_size_h, w, patch_size_w]
        patch_original = patch_original.reshape(N, LL * time_patch_size, 3, HH * patch_size, WW * patch_size)  # [N, L_all, C, H, W]
        
        patch_pred = patch_pred.reshape(N, *macro_patch_grids, channels, *patch_size_grids)  # [N, LL, HH, WW, C, l, h, w]
        patch_pred = patch_pred.expand(-1, -1, -1, -1, 3, -1, -1, -1)
        patch_pred = patch_pred.permute(0, 1, 5, 4, 2, 6, 3, 7)  # [N, LL, time_patch_size, C, h, patch_size_h, w, patch_size_w]
        patch_pred = patch_pred.reshape(N, LL * time_patch_size, 3, HH * patch_size, WW * patch_size)  # [N, L_all, C, H, W]
        
        patch_mask_color = patch_mask_color.reshape(N, *macro_patch_grids, 3, *patch_size_grids)  # [N, LL, HH, WW, C, l, h, w]
        patch_mask_color = patch_mask_color.permute(0, 1, 5, 4, 2, 6, 3, 7)  # [N, LL, time_patch_size, C, h, patch_size_h, w, patch_size_w]
        patch_mask_color = patch_mask_color.reshape(N, LL * time_patch_size, 3, HH * patch_size, WW * patch_size)  # [N, L_all, C, H, W]
        
        save_dir = os.path.join(self.infer_save_dir, 'inference_videos')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # restore values by mean and std
        patch_original = patch_original * std + mean  # 0~255
        patch_pred = patch_pred * std + mean  # 0~255
        patch_mask_color = patch_mask_color * std + mean  # 0~255
        
        # [N, L, 3, H, W] -> [N, 3, L, H, W]
        patch_original = patch_original.detach().clamp(0, 255).byte().transpose(1, 2).cpu().numpy()  # 0~255 uint8
        patch_pred = patch_pred.detach().clamp(0, 255).byte().transpose(1, 2).cpu().numpy()  # 0~255 uint8
        patch_mask_color = patch_mask_color.detach().clamp(0, 255).byte().transpose(1, 2).cpu().numpy()  # 0~255 uint8
        
        # up = np.concatenate([patch_pred, patch_original], axis=4)
        # down = np.concatenate([patch_mask_color, np.zeros_like(patch_original)], axis=4)
        # all = np.concatenate([up, down], axis=3)
        all = np.concatenate([patch_original, patch_mask_color, patch_pred], axis=3)  # [N, 3, L_all, H_cat, W]
        
        if save_as_video:
            # save video
            from src.datasets.modules.media_rw import save_video
            fps = 10
            for n_idx in range(patch_original.shape[0]):  # loop for batch_size
                output_path = os.path.join(save_dir, file_name[n_idx] + '.avi')
                save_video(all[n_idx], output_path, fps)
        else:
            # save images
            from PIL import Image
            for n_idx in range(patch_original.shape[0]):
                print(file_name[n_idx])
                os.makedirs(os.path.join(save_dir, file_name[n_idx]))
                for l_idx in range(patch_original.shape[2]):
                    output_path = os.path.join(save_dir, file_name[n_idx], f'{l_idx}.png')
                    Image.fromarray(all[n_idx, :, l_idx].transpose(1, 2, 0)).save(output_path)