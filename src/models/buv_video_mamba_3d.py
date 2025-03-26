import torch
from torch import nn

from .modules.basic_layers import MLP
from .modules.echo_dynamic_head import EchoDynamicEFHead
from .modules.model_base import ModelBase, model_register


@model_register('buv_video_mamba_3d')
class BuvVideoMamba3DModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        assert getattr(cfg.model, 'new_scan', False)
        from .modules.video_mamba_3d_new import VideoMamba3D, VideoMamba3DFinal
        
        self.net = VideoMamba3D(cfg)
        
        # if getattr(cfg.model, 'new_scan', False):
        #     self.extra = VideoMamba3DExtra(
        #         self.net.encoder[-1],
        #         extra_blocks_on_global=getattr(cfg.model, 'extra_blocks_on_global', 0),
        #         mamba_cls=self.net.mamba_cls,
        #         n_mamba_per_block=cfg.model.n_mamba_per_block,
        #         mamba_3d_forward_type=getattr(cfg.model, 'mamba_3d_forward_type', 'serial'),
        #         )
        # else:
        #     self.extra = nn.Identity()
            
        self.final = VideoMamba3DFinal(
            cfg,
            self.net.encoder[-1],
            )
        self.frame_final = BuvFrameLevelFinal(cfg, self.net.encoder[-1])
        
        self.cls_head = EchoDynamicEFHead(cfg, self.final.final_dim)
        self.frame_cls_head = EchoDynamicEFHead(cfg, self.frame_final.final_dim)
    
    def forward(self, inputs: dict) -> dict:
        x = inputs['x']  # [N, C, L_all, H, W]

        x, x_2d = self.net(x, additional_2d_only=True)  # [N, LLLL, HHHH, WWWW, D]
        # x = self.extra(
        #     x,
        #     self.net.l_reg_token_indices,
        #     self.net.h_reg_token_indices,
        #     self.net.w_reg_token_indices,
        #     )  # [N, LLLL, HHHH, WWWW, D]
        
        ## XXX
        # x_2d & x_frame is not used as frame_loss_weight is 0.
        x_frame = self.frame_final(
            x_2d,
            self.net.l_reg_token_indices,
            self.net.h_reg_token_indices,
            self.net.w_reg_token_indices,
            )  # [N, L, final_dim] or [N, L, n_global_frame, final_dim]
        x_frame = self.frame_cls_head(x_frame)  # [N, L, 1] or [N, L, n_global_frame, 1]
        
        x = self.final(
            x,
            self.net.l_reg_token_indices,
            self.net.h_reg_token_indices,
            self.net.w_reg_token_indices,
            )  # [N, final_dim] or [N, n_global, final_dim]
        x = self.cls_head(x)  # [N, 1] or [N, n_global, 1]
        
        return {
            'pred_y': x,  # [N, 1] or [N, n_global, 1]
            'pred_frame_y': x_frame,  # [N, L, 1] or [N, L, n_global_frame, 1]
        }


class BuvFrameLevelFinal(nn.Module):
    def __init__(self, cfg, one_block_sample):
        super().__init__()
        
        self.final_dim = getattr(cfg.model, 'final_dim', one_block_sample.embed_dim)
        
        self.final_aggregate_mode = getattr(cfg.model, 'final_aggregate_mode', 'all_mean')
        if self.final_aggregate_mode == 'all_enclosure_mean':
            self.final_aggregate_mode = 'all_mean'
            
        if self.final_aggregate_mode in ['all_mean', 'all_max', 'none']:
            self.final_linear = MLP(one_block_sample.embed_dim, [self.final_dim], nn.SiLU, final_activation=True)
            
        elif self.final_aggregate_mode == 'all_concat':
            num_used_global = (2 + getattr(cfg.model, 'h_reg_num', 0)) * (2 + getattr(cfg.model, 'w_reg_num', 0))
            self.final_linear = MLP(num_used_global * one_block_sample.embed_dim, [self.final_dim], nn.SiLU, final_activation=True)
        
        else:
            raise ValueError(f'final_aggregate_mode {self.final_aggregate_mode} not supported')
        
    def forward(self, x: torch.Tensor, l_reg_token_indices, h_reg_token_indices, w_reg_token_indices):  # [N, LLL, HHH, WWW, D]
        LLL = x.shape[1]
        l_real_mask = torch.ones(LLL, dtype=torch.bool, device=x.device)
        l_real_mask[l_reg_token_indices] = False
        x = x[:, l_real_mask][:, :, h_reg_token_indices][:, :, :, w_reg_token_indices]  # [N, L, H_global, W_global, D]
        x = x.flatten(2, -2)  # [N, L, n_global, D]
        
        N, L, n_global = x.shape[:3]
        x = x.flatten(0, 1)  # [N*L, n_global, D]
        
        if self.final_aggregate_mode == 'all_mean':
            x = x.mean(dim=1)  # [N*L, D]
        elif self.final_aggregate_mode == 'all_max':
            x = x.max(dim=1).values  # [N*L, D]
        elif self.final_aggregate_mode == 'all_concat':
            x = x.flatten(1)  # [N*L, n_global * D]
        elif self.final_aggregate_mode == 'none':
            x = x.reshape(N, L, n_global, -1)  # [N, L, n_global, D]
            x = self.final_linear(x)  # [N, L, n_global, final_dim]
            return x  # [N, L, n_global, final_dim]
        
        x = self.final_linear(x)
        x = x.reshape(N, L, -1)  # [N, L, final_dim]
        return x