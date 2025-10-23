import torch
from torch import nn

from src.datasets.camus_dataset import _EF_MEAN_OFFSET as _EF_MEAN_OFFSET_CAMUS
from src.datasets.echo_dynamic_dataset import _EF_MEAN_OFFSET

from .modules.echo_dynamic_head import EchoDynamicClsHead, EchoDynamicEFHead
from .modules.model_base import ModelBase, model_register


@model_register('echo_dynamic_video_mamba_3d_seg')
class EchoDynamicVideoMamba3DSegModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        if getattr(cfg.model, 'new_scan', False):
            from .modules.video_mamba_3d_new import (VideoMamba3D,
                                                     VideoMamba3DFinal)
        else:
            from .modules.video_mamba_3d_old import (VideoMamba3D,
                                                     VideoMamba3DFinal)
            
        self.net = VideoMamba3D(cfg)
            
        self.final = VideoMamba3DSegFinal(
            hidden_dim=self.net.embed_dim,
            out_channels=2,
            time_patch_size=self.net.time_patch_size,
            patch_size=self.net.patch_size,
            )
    
    def forward(self, inputs: dict) -> dict:
        if getattr(self.cfg.model, 'save_delta', False):
            self._save_input_for_delta_vis(inputs, verbose=True)
        
        x = inputs['x']  # [N, C, L_all, H, W] or [N, clips, C, L_all, H, W]
        if x.dim() == 6:
            N, clips, C, L_all, H, W = x.shape
            x = x.reshape(N * clips, C, L_all, H, W)
        else:
            N, C, L_all, H, W = x.shape
            clips = 1

        x = self.net(x)  # [N, LLLL, HHHH, WWWW, D]
        
        x = self.final(
            x,
            real_tokens_mask=self.net.real_tokens_mask.expand(N, -1, -1, -1),
            grid_size=self.net.patch_embed.grid_size,
            )  # [N, embed_dim]
        
        return {
            'pred_seg': x,
        }


class VideoMamba3DSegFinal(nn.Module):
    def __init__(self, hidden_dim, out_channels, time_patch_size, patch_size):
        super().__init__()
        self.out_channels = out_channels
        self.patch_size = (time_patch_size, patch_size, patch_size)

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.patch_reconstruct = nn.Linear(hidden_dim, out_channels * time_patch_size * patch_size * patch_size)
        
    def _restore_3d_structure(self, patch_pred):  # [N, LL, HH, WW, C*D_patch_size*H_patch_size*W_patch_size]
        N, LL, HH, WW, _ = patch_pred.shape
        patch_pred = patch_pred.reshape(N, LL, HH, WW, self.out_channels, *self.patch_size)  # [N, LL, HH, WW, C, l, h, w]
        patch_pred = patch_pred.permute(0, 4, 1, 5, 2, 6, 3, 7)  # [N, C, LL, D_patch_size, h, H_patch_size, w, W_patch_size]
        patch_pred = patch_pred.reshape(N, self.out_channels, LL * self.patch_size[0], HH * self.patch_size[1], WW * self.patch_size[2])  # [N, C, L_all, H, W]
        return patch_pred
        
    def forward(self, x, real_tokens_mask, grid_size):  # [N, L, H, W, D] -> [N, L_all, num_patches_per_frame, patch_channels_all]
        x = self.final_norm(x)  # [N, L, H, W, D]
                
        N, D = x.shape[0], x.shape[-1]
        x = x[real_tokens_mask].reshape(N, *grid_size, D)  # [N, LL, HH, WW, D]
        
        x = self.patch_reconstruct(x)  # [N, LL, HH, WW, C*time_patch_size*patch_size*patch_size]
        x = self._restore_3d_structure(x)  # [N, C, L_all, H, W]
        return x