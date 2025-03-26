from torch import nn

from src.datasets.camus_dataset import _EF_MEAN_OFFSET as _EF_MEAN_OFFSET_CAMUS
from src.datasets.echo_dynamic_dataset import _EF_MEAN_OFFSET

from .modules.echo_dynamic_head import EchoDynamicEFHead
from .modules.model_base import ModelBase, model_register


@model_register('echo_dynamic_video_mamba_3d')
class EchoDynamicVideoMamba3DModel(ModelBase):
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
        
        self.head_type = getattr(cfg.model, 'head_type', 'default')
        if self.head_type == 'default':  # for EchoNet-Dynamic EF regression
            self.ef_head = EchoDynamicEFHead(cfg, self.final.final_dim, bias_init=_EF_MEAN_OFFSET)
        elif self.head_type == 'camus_ef':  # for CAMUS EF regression
            self.ef_head = EchoDynamicEFHead(cfg, self.final.final_dim, bias_init=_EF_MEAN_OFFSET_CAMUS)
        else:
            raise NotImplementedError(f'head_type "{self.head_type}" has not been implemented yet.')
    
    def forward(self, inputs: dict) -> dict:
        x = inputs['x']  # [N, C, L_all, H, W] or [N, clips, C, L_all, H, W]
        if x.dim() == 6:
            N, clips, C, L_all, H, W = x.shape
            x = x.reshape(N * clips, C, L_all, H, W)
        else:
            N, C, L_all, H, W = x.shape
            clips = 1

        x = self.net(x)  # [N, LLLL, HHHH, WWWW, D]
        # x = self.extra(
        #     x,
        #     self.net.l_reg_token_indices,
        #     self.net.h_reg_token_indices,
        #     self.net.w_reg_token_indices,
        #     )  # [N, LLLL, HHHH, WWWW, D]
        
        x = self.final(
            x,
            self.net.l_reg_token_indices,
            self.net.h_reg_token_indices,
            self.net.w_reg_token_indices,
            )  # [N, embed_dim]
        
        if self.head_type in ['default', 'camus_ef']:
            x = self.ef_head(x)
            return_name = 'pred_ef'
        else:
            raise NotImplementedError(f'head_type "{self.head_type}" has not been implemented yet.')
        
        x = x.reshape(N, clips, 1)
        x = x.mean(dim=1)  # [N, 1]
        
        return {
            return_name: x,
        }
