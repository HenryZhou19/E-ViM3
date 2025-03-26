import os
from functools import partial
from typing import List

import numpy as np
import torch
from torch import nn

from src.utils.misc import DistMisc

from .basic_layers import MLP, PatchEmbedding3D
from .mamba import Mamba, Mamba2Block, MambaBlock

os.environ['TRITON_LIBCUDA_PATH'] = '/usr/local/cuda/lib64/stubs/'

class VideoMamba3D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self._get_patch_embed()
        self._get_enclosure_tokens()
        self._get_pos_embed()
        self._get_mamba()
        
        self.l_reg_token_indices: List[int] = None
        self.h_reg_token_indices: List[int] = None
        self.w_reg_token_indices: List[int] = None
        self.real_tokens_mask:torch.Tensor = None
        
    def _get_patch_embed(self):
        self.in_channels = self.cfg.data.in_channels
        self.img_size = self.cfg.data.resize_to
        self.video_length = self.cfg.data.length
        self.time_patch_size = getattr(self.cfg.model, 'time_patch_size', 1)
        self.patch_size = getattr(self.cfg.model, 'patch_size', 16)
        self.embed_dim = getattr(self.cfg.model, 'embed_dim', 384)
        
        self.patch_embed = PatchEmbedding3D(
            tensor_dhw=(self.video_length, self.img_size, self.img_size),
            patch_size=(self.time_patch_size, self.patch_size, self.patch_size),
            stride=(self.time_patch_size, self.patch_size, self.patch_size),
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            norm_layer=nn.Identity,
            flatten=False,
            )
        
    def _get_pos_embed(self):
        self.l_pos_embed = nn.Parameter(torch.randn(1, 2 + self.l_reg_num + self.patch_embed.grid_size[0], 1, 1, self.embed_dim) * 0.02)
        self.h_pos_embed = nn.Parameter(torch.randn(1, 1, 2 + self.h_reg_num + self.patch_embed.grid_size[1], 1, self.embed_dim) * 0.02)
        self.w_pos_embed = nn.Parameter(torch.randn(1, 1, 1, 2 + self.w_reg_num + self.patch_embed.grid_size[2], self.embed_dim) * 0.02)
        
    def _get_enclosure_tokens(self):
        # 6 faces of the video
        self.global_faces_token = nn.Parameter(torch.zeros(6, 1, 1, 1, 1, self.embed_dim))
        # 12 edges of the video
        self.global_edges_token = nn.Parameter(torch.zeros(12, 1, 1, 1, 1, self.embed_dim))
        # 8 corners of the video
        self.global_corners_token = nn.Parameter(torch.zeros(8, 1, 1, 1, 1, self.embed_dim))
        # 1 global register token
        if getattr(self.cfg.model, 'unified_reg_num', None) is not None:
            self.l_reg_num = self.h_reg_num = self.w_reg_num = self.cfg.model.unified_reg_num
        else:
            self.l_reg_num = getattr(self.cfg.model, 'l_reg_num', 0)
            self.h_reg_num = getattr(self.cfg.model, 'h_reg_num', 0)
            self.w_reg_num = getattr(self.cfg.model, 'w_reg_num', 0)
        if self.l_reg_num + self.h_reg_num + self.w_reg_num > 0:
            self.global_reg_token = nn.Parameter(torch.zeros(1, 1, 1, 1, self.embed_dim))
        else:
            self.register_buffer('global_reg_token', torch.zeros(1, 1, 1, 1, self.embed_dim), persistent=False)
        
    def _get_mamba(self):
        self.macro_block_num = getattr(self.cfg.model, 'macro_block_num', 12)
        
        if self.cfg.model.mamba_block_type == 'mamba':
            micro_mamba_block_cls = MambaBlock
            micro_mamba_block_config = {}  # d_model will be passed in Mamba
        elif self.cfg.model.mamba_block_type == 'mamba2':
            micro_mamba_block_cls = Mamba2Block
            # n_head = 8
            assert self.embed_dim * 2 % 8 == 0, f'embed_dim * expand(=2) should be divided by n_head(=8), got {self.embed_dim}'
            micro_mamba_block_config = {
                'headdim': self.embed_dim * 2 // 8,
                'no_triton': False,
                }
        
        self.mamba_cls = partial(Mamba,
            mamba_block_cls=micro_mamba_block_cls,
            mamba_block_config=micro_mamba_block_config,
            rms_norm=False,
            residual_in_fp32=True,
            no_amp=self.cfg.env.amp.amp_mode == 'fp16',
            final_norm=False,
            fused_add_norm=True,  # XXX
            drop_path=getattr(self.cfg.model, 'drop_path', 0.0),
            )
        
        forward_type = getattr(self.cfg.model, 'mamba_3d_forward_type', 'serial')
        mamba_layer_per_macro = 6 if forward_type == 'serial' else 1
        self.encoder = nn.ModuleList([
            Mamba3D(self.mamba_cls, self.embed_dim, self.cfg.model.n_mamba_per_block, total_mamba_blocks=self.macro_block_num * self.cfg.model.n_mamba_per_block * mamba_layer_per_macro, forward_type=forward_type)
            for _ in range(self.macro_block_num)
            ])
    
    def _add_enclosure_tokens(self, x: torch.Tensor) -> torch.Tensor:
        N, LL, HH, WW, D = x.shape
        LLL, HHH, WWW = LL + self.l_reg_num, HH + self.h_reg_num, WW + self.w_reg_num
        
        x_original = x
        x = torch.empty(
            N,
            LLL + 2,
            HHH + 2,
            WWW + 2,
            D,
            device=x.device, dtype=x.dtype)
        
        # fill all (the remaining at last) with gloabl register token
        x[:] = self.global_reg_token
        
        # fill the faces with the global faces token, edges with the global edges token, and corners with the global corners token
        x[:, :1, 1:-1, 1:-1, :] = self.global_faces_token[0].expand(N, 1, HHH, WWW, D)
        x[:, -1:, 1:-1, 1:-1, :] = self.global_faces_token[1].expand(N, 1, HHH, WWW, D)
        x[:, 1:-1, :1, 1:-1, :] = self.global_faces_token[2].expand(N, LLL, 1, WWW, D)
        x[:, 1:-1, -1:, 1:-1, :] = self.global_faces_token[3].expand(N, LLL, 1, WWW, D)
        x[:, 1:-1, 1:-1, :1, :] = self.global_faces_token[4].expand(N, LLL, HHH, 1, D)
        x[:, 1:-1, 1:-1, -1:, :] = self.global_faces_token[5].expand(N, LLL, HHH, 1, D)
        
        x[:, :1, :1, 1:-1, :] = self.global_edges_token[0].expand(N, 1, 1, WWW, D)
        x[:, -1:, :1, 1:-1, :] = self.global_edges_token[1].expand(N, 1, 1, WWW, D)
        x[:, :1, -1:, 1:-1, :] = self.global_edges_token[2].expand(N, 1, 1, WWW, D)
        x[:, -1:, -1:, 1:-1, :] = self.global_edges_token[3].expand(N, 1, 1, WWW, D)
        x[:, :1, 1:-1, :1, :] = self.global_edges_token[4].expand(N, 1, HHH, 1, D)
        x[:, -1:, 1:-1, :1, :] = self.global_edges_token[5].expand(N, 1, HHH, 1, D)
        x[:, :1, 1:-1, -1:, :] = self.global_edges_token[6].expand(N, 1, HHH, 1, D)
        x[:, -1:, 1:-1, -1:, :] = self.global_edges_token[7].expand(N, 1, HHH, 1, D)
        x[:, 1:-1, :1, :1, :] = self.global_edges_token[8].expand(N, LLL, 1, 1, D)
        x[:, 1:-1, -1:, :1, :] = self.global_edges_token[9].expand(N, LLL, 1, 1, D)
        x[:, 1:-1, :1, -1:, :] = self.global_edges_token[10].expand(N, LLL, 1, 1, D)
        x[:, 1:-1, -1:, -1:, :] = self.global_edges_token[11].expand(N, LLL, 1, 1, D)
        
        x[:, :1, :1, :1, :] = self.global_corners_token[0].expand(N, 1, 1, 1, D)
        x[:, -1:, :1, :1, :] = self.global_corners_token[1].expand(N, 1, 1, 1, D)
        x[:, :1, -1:, :1, :] = self.global_corners_token[2].expand(N, 1, 1, 1, D)
        x[:, -1:, -1:, :1, :] = self.global_corners_token[3].expand(N, 1, 1, 1, D)
        x[:, :1, :1, -1:, :] = self.global_corners_token[4].expand(N, 1, 1, 1, D)
        x[:, -1:, :1, -1:, :] = self.global_corners_token[5].expand(N, 1, 1, 1, D)
        x[:, :1, -1:, -1:, :] = self.global_corners_token[6].expand(N, 1, 1, 1, D)
        x[:, -1:, -1:, -1:, :] = self.global_corners_token[7].expand(N, 1, 1, 1, D)
        
        if self.real_tokens_mask is None or self.real_tokens_mask.shape[1:] != x.shape[1:-1]:
            l_global_num, h_global_num, w_global_num = 2 + self.l_reg_num, 2 + self.h_reg_num, 2 + self.w_reg_num
            print(f'Rank {DistMisc.get_rank()} - Recalculating real tokens mask...', force=True)
            self.l_reg_token_indices: List[int] = np.linspace(0, LLL + 1, l_global_num, dtype=np.int64).tolist()
            self.h_reg_token_indices: List[int] = np.linspace(0, HHH + 1, h_global_num, dtype=np.int64).tolist()
            self.w_reg_token_indices: List[int] = np.linspace(0, WWW + 1, w_global_num, dtype=np.int64).tolist()
            self.real_tokens_mask = get_real_tokens_mask(x, self.l_reg_token_indices, self.h_reg_token_indices, self.w_reg_token_indices)
        
        x[self.real_tokens_mask.expand(N, -1, -1, -1)] = x_original.flatten(0, -2)
        
        return x
        
    def forward(self, x: torch.Tensor, external_patch_embed=False, patch_mask=None) -> torch.Tensor:  # [N, C, L, H, W]
        if external_patch_embed:
            pass
        else:
            x = self.patch_embed(x)  # [N, LL, HH, WW, D]
        
        x = self._add_enclosure_tokens(x)
        
        N = x.shape[0]
        x = x + self.l_pos_embed.expand(N, -1, -1, -1, -1)
        x = x + self.h_pos_embed.expand(N, -1, -1, -1, -1)
        x = x + self.w_pos_embed.expand(N, -1, -1, -1, -1)
        
        if patch_mask is not None:
            no_mask = torch.ones(*x.shape[0:4], dtype=torch.bool, device=x.device)
            no_mask[self.real_tokens_mask.expand(N, -1, -1, -1)] = torch.logical_not(patch_mask).flatten()  # mask: True for meanful positions
            no_mask_arange = torch.arange(no_mask.sum(), device=x.device)
            
            x_lhw_real = x[no_mask].reshape(N, -1, self.embed_dim)
            
            no_mask_lhw_order = -torch.ones(*x.shape[0:4], dtype=torch.long, device=x.device)
            no_mask_lhw_order[no_mask] = no_mask_arange
            no_mask_lhw_order = no_mask_lhw_order.permute(0, 2, 3, 1)  # [N, H, W, L]
            no_mask = no_mask.permute(0, 2, 3, 1)  # [N, H, W, L]
            lhw_2_hwl = no_mask_lhw_order[no_mask]  # [N*H*W*L]
            
            no_mask_hwl_order = no_mask_lhw_order
            no_mask_hwl_order[no_mask] = no_mask_arange
            no_mask_hwl_order = no_mask_hwl_order.permute(0, 3, 2, 1)  # [N, L, W, H]
            no_mask = no_mask.permute(0, 3, 2, 1)  # [N, L, W, H]
            hwl_2_lwh = no_mask_hwl_order[no_mask]  # [N*L*W*H]
            
            no_mask_lwh_order = no_mask_hwl_order
            no_mask_lwh_order[no_mask] = no_mask_arange
            no_mask_lwh_order = no_mask_lwh_order.permute(0, 1, 3, 2)  # [N, L, H, W]
            no_mask = no_mask.permute(0, 1, 3, 2)  # [N, L, H, W]
            lwh_2_lhw = no_mask_lwh_order[no_mask]  # [N*L*H*W]
            
            for enc in self.encoder:
                x_lhw_real = enc(x_lhw_real, (lhw_2_hwl, hwl_2_lwh, lwh_2_lhw))
            x[no_mask] = x_lhw_real.flatten(0, 1)
        else:
            for enc in self.encoder:
                x = enc(x)
        return x
 
        
class Mamba3D(nn.Module):
    def __init__(self, mamba_block_cls, embed_dim, n_mamba_per_block, total_mamba_blocks, forward_type='serial'):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.hwl_mamba = mamba_block_cls(d_model=embed_dim, n_layer=n_mamba_per_block, total_layers=total_mamba_blocks)
        self.r_hwl_mamba = mamba_block_cls(d_model=embed_dim, n_layer=n_mamba_per_block, total_layers=total_mamba_blocks)
        
        self.lwh_mamba = mamba_block_cls(d_model=embed_dim, n_layer=n_mamba_per_block, total_layers=total_mamba_blocks)
        self.r_lwh_mamba = mamba_block_cls(d_model=embed_dim, n_layer=n_mamba_per_block, total_layers=total_mamba_blocks)
        
        self.lhw_mamba = mamba_block_cls(d_model=embed_dim, n_layer=n_mamba_per_block, total_layers=total_mamba_blocks)
        self.r_lhw_mamba = mamba_block_cls(d_model=embed_dim, n_layer=n_mamba_per_block, total_layers=total_mamba_blocks)
        
        assert forward_type == 'serial', f'forward_type should be "serial", but got {forward_type}'
        
    def forward(self, x, orders=None):  # [N, L, H, W, D]
        if orders is None:
            N, L, H, W, D = x.shape
            
            x = x.permute(0, 2, 3, 1, 4)  # [N, H, W, L, D]
            x = x.flatten(1, -2)  # [N, H * W * L, D]
            x = self.hwl_mamba(x)
            x = self.r_hwl_mamba(x.flip(1)).flip(1)
            x = x.reshape(N, H, W, L, D)  # [N, H, W, L, D]
            
            x = x.permute(0, 3, 2, 1, 4)  # [N, L, W, H, D]
            x = x.flatten(1, -2)  # [N, L * W * H, D]
            x = self.lwh_mamba(x)
            x = self.r_lwh_mamba(x.flip(1)).flip(1)
            x = x.reshape(N, L, W, H, D)  # [N, L, W, H, D]
            
            x = x.permute(0, 1, 3, 2, 4)  # [N, L, H, W, D]
            x = x.flatten(1, -2)  # [N, L * H * W, D]
            x = self.lhw_mamba(x)
            x = self.r_lhw_mamba(x.flip(1)).flip(1)
            x = x.reshape(N, L, H, W, D)  # [N, L, H, W, D]
        else:
            lhw_2_hwl, hwl_2_lwh, lwh_2_lhw = orders
            # x [N, LLL, D] in lhw_order
            N, LLL, D = x.shape
            
            x = x.flatten(0, 1)[lhw_2_hwl].reshape(N, LLL, D)  # [N, LLL, D] in hwl_order
            x = self.hwl_mamba(x)
            x = self.r_hwl_mamba(x.flip(1)).flip(1)
            
            x = x.flatten(0, 1)[hwl_2_lwh].reshape(N, LLL, D)  # [N, LLL, D] in lwh_order
            x = self.lwh_mamba(x)
            x = self.r_lwh_mamba(x.flip(1)).flip(1)
            
            x = x.flatten(0, 1)[lwh_2_lhw].reshape(N, LLL, D)  # [N, LLL, D] in lhw_order
            x = self.lhw_mamba(x)
            x = self.r_lhw_mamba(x.flip(1)).flip(1)
        
        return x
    
    
class VideoMamba3DDecoderForPretraining(nn.Module):
    def __init__(self, additional_blocks: int, mamba_cls: Mamba, n_mamba_per_block, hidden_dim, out_channels, time_patch_size, patch_size, mamba_3d_forward_type):
        super().__init__()
        self.additional_blocks = additional_blocks
        if self.additional_blocks > 0:
            mamba_layer_per_macro = 6 if mamba_3d_forward_type == 'serial' else 1
            self.decoder_blocks = nn.Sequential(*[
                Mamba3D(mamba_cls, hidden_dim, n_mamba_per_block, total_mamba_blocks=self.additional_blocks * n_mamba_per_block * mamba_layer_per_macro, forward_type=mamba_3d_forward_type)
                for _ in range(self.additional_blocks)
                ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.patch_reconstruct = nn.Linear(hidden_dim, out_channels * time_patch_size * patch_size * patch_size)
        
    def forward(self, x, real_tokens_mask, grid_size):  # [N, L, H, W, D] -> [N, L_all, num_patches_per_frame, patch_channels_all]
        if self.additional_blocks > 0:
            x = self.decoder_blocks(x)  # [N, L, H, W, D]
        x = self.final_norm(x)  # [N, L, H, W, D]
        
        N, D = x.shape[0], x.shape[-1]
        x = x[real_tokens_mask].reshape(N, *grid_size, D)  # [N, LL, HH, WW, D]
        x = self.patch_reconstruct(x)  # [N, LL, HH, WW, C*time_patch_size*patch_size*patch_size]
        return x
    
'''
efficient version is only for pre-training
so we don't need to consider the final aggregation mode here
(the real one is in 'video_mamba_3d_new')
'''
# class VideoMamba3DFinal(nn.Module):
#     pass

def get_real_tokens_mask(x: torch.Tensor, l_reg_token_indices: List[int], h_reg_token_indices: List[int], w_reg_token_indices: List[int]):
    _, LL, HH, WW, _ = x.shape
    
    l_reg_tokens_mask = torch.zeros(LL, dtype=torch.bool, device=x.device)
    l_reg_tokens_mask[l_reg_token_indices] = True
    
    h_reg_tokens_mask = torch.zeros(HH, dtype=torch.bool, device=x.device)
    h_reg_tokens_mask[h_reg_token_indices] = True
    
    w_reg_tokens_mask = torch.zeros(WW, dtype=torch.bool, device=x.device)
    w_reg_tokens_mask[w_reg_token_indices] = True
    
    mask = torch.ones(1, *x.shape[1:-1], dtype=torch.bool, device=x.device)
    mask[:, l_reg_tokens_mask, :, :] = False
    mask[:, :, h_reg_tokens_mask, :] = False
    mask[:, :, :, w_reg_tokens_mask] = False
    
    return mask