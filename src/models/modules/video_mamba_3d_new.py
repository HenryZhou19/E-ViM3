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
        if getattr(self.cfg.model, 'use_pos_embed', True):
            self.l_pos_embed = nn.Parameter(torch.randn(1, 2 + self.l_reg_num + self.patch_embed.grid_size[0], 1, 1, self.embed_dim) * 0.02)
            self.h_pos_embed = nn.Parameter(torch.randn(1, 1, 2 + self.h_reg_num + self.patch_embed.grid_size[1], 1, self.embed_dim) * 0.02)
            self.w_pos_embed = nn.Parameter(torch.randn(1, 1, 1, 2 + self.w_reg_num + self.patch_embed.grid_size[2], self.embed_dim) * 0.02)
        else:
            l_pos_embed = torch.zeros(1, 2 + self.l_reg_num + self.patch_embed.grid_size[0], 1, 1, self.embed_dim)
            self.register_buffer('l_pos_embed', l_pos_embed)
            h_pos_embed = torch.zeros(1, 1, 2 + self.h_reg_num + self.patch_embed.grid_size[1], 1, self.embed_dim)
            self.register_buffer('h_pos_embed', h_pos_embed)
            w_pos_embed = torch.zeros(1, 1, 1, 2 + self.w_reg_num + self.patch_embed.grid_size[2], self.embed_dim)
            self.register_buffer('w_pos_embed', w_pos_embed)
            print('\n\n\nNot using pos_embed!\n\n\n', force=True)
        
    def _get_enclosure_tokens(self):
        enclosure_type = getattr(self.cfg.model, 'enclosure_type', 'default')
        if enclosure_type == 'default':
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
        elif enclosure_type == 'all_unified':
            self.global_token = nn.Parameter(torch.zeros(1, 1, 1, 1, self.embed_dim))
            self.global_faces_token = self.global_token.unsqueeze(0).expand(6, -1, -1, -1, -1, -1)
            self.global_edges_token = self.global_token.unsqueeze(0).expand(12, -1, -1, -1, -1, -1)
            self.global_corners_token = self.global_token.unsqueeze(0).expand(8, -1, -1, -1, -1, -1)
            if getattr(self.cfg.model, 'unified_reg_num', None) is not None:
                self.l_reg_num = self.h_reg_num = self.w_reg_num = self.cfg.model.unified_reg_num
            else:
                self.l_reg_num = getattr(self.cfg.model, 'l_reg_num', 0)
                self.h_reg_num = getattr(self.cfg.model, 'h_reg_num', 0)
                self.w_reg_num = getattr(self.cfg.model, 'w_reg_num', 0)
            if self.l_reg_num + self.h_reg_num + self.w_reg_num > 0:
                self.global_reg_token = self.global_token
            else:
                self.register_buffer('global_reg_token', torch.zeros(1, 1, 1, 1, self.embed_dim), persistent=False)
        elif enclosure_type == 'group_unified':
            self.global_token1 = nn.Parameter(torch.zeros(1, 1, 1, 1, self.embed_dim))
            self.global_faces_token = self.global_token1.unsqueeze(0).expand(6, -1, -1, -1, -1, -1)
            self.global_token2 = nn.Parameter(torch.zeros(1, 1, 1, 1, self.embed_dim))
            self.global_edges_token = self.global_token2.unsqueeze(0).expand(12, -1, -1, -1, -1, -1)
            self.global_token3 = nn.Parameter(torch.zeros(1, 1, 1, 1, self.embed_dim))
            self.global_corners_token = self.global_token3.unsqueeze(0).expand(8, -1, -1, -1, -1, -1)
            if getattr(self.cfg.model, 'unified_reg_num', None) is not None:
                self.l_reg_num = self.h_reg_num = self.w_reg_num = self.cfg.model.unified_reg_num
            else:
                self.l_reg_num = getattr(self.cfg.model, 'l_reg_num', 0)
                self.h_reg_num = getattr(self.cfg.model, 'h_reg_num', 0)
                self.w_reg_num = getattr(self.cfg.model, 'w_reg_num', 0)
            if self.l_reg_num + self.h_reg_num + self.w_reg_num > 0:
                self.global_token4 = nn.Parameter(torch.zeros(1, 1, 1, 1, self.embed_dim))
                self.global_reg_token = self.global_token4
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
            drop_path=getattr(self.cfg.model, 'drop_path', 0.0)
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
        
    def forward(self, x: torch.Tensor, external_patch_embed=False, additional_2d_only=False) -> torch.Tensor:  # [N, C, L, H, W]
        if external_patch_embed:
            pass
        else:
            x = self.patch_embed(x)  # [N, LL, HH, WW, D]
        
        x = self._add_enclosure_tokens(x)
        
        N = x.shape[0]
        x = x + self.l_pos_embed.expand(N, -1, -1, -1, -1)
        x = x + self.h_pos_embed.expand(N, -1, -1, -1, -1)
        x = x + self.w_pos_embed.expand(N, -1, -1, -1, -1)
        
        if additional_2d_only:
            x_2d = x.clone()
            x_2d = x_2d.flatten(0, 1)  # [N * LLLL, HHHH, WWWW, D]  
            for enc in self.encoder:
                x_2d = enc.forward_2d_only(x_2d)
                x = enc(x)  # [N, LLLL, HHHH, WWWW, D]
            x_2d = x_2d.reshape(N, -1, *x_2d.shape[1:])  # [N, LLLL, HHHH, WWWW, D]
            return x, x_2d
        else:
            for enc in self.encoder:
                x = enc(x)  # [N, LLLL, HHHH, WWWW, D]
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
        
    def forward(self, x: torch.Tensor):  # [N, L, H, W, D]
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
        
        return x
    
    def forward_2d_only(self, x: torch.Tensor):  # [NN, H, W, D]
        NN, H, W, D = x.shape
        
        x = x.permute(0, 2, 1, 3)  # [NN, W, H, D]
        x = x.flatten(1, -2)  # [NN, W * H, D]
        x = self.lwh_mamba(x)
        x = self.r_lwh_mamba(x.flip(1)).flip(1)
        x = x.reshape(NN, W, H, D)  # [NN, W, H, D]
        
        x = x.permute(0, 2, 1, 3)  # [NN, H, W, D]
        x = x.flatten(1, -2)  # [NN, H * W, D]
        x = self.lhw_mamba(x)
        x = self.r_lhw_mamba(x.flip(1)).flip(1)
        x = x.reshape(NN, H, W, D)
        
        return x
    
class Mamba3DExtra(nn.Module):
    def __init__(self, mamba_block_cls, embed_dim, n_mamba_per_block, total_mamba_blocks, forward_type='serial'):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.l_mamba = mamba_block_cls(d_model=embed_dim, n_layer=n_mamba_per_block, total_layers=total_mamba_blocks)
        self.r_l_mamba = mamba_block_cls(d_model=embed_dim, n_layer=n_mamba_per_block, total_layers=total_mamba_blocks)
        
        self.h_mamba = mamba_block_cls(d_model=embed_dim, n_layer=n_mamba_per_block, total_layers=total_mamba_blocks)
        self.r_h_mamba = mamba_block_cls(d_model=embed_dim, n_layer=n_mamba_per_block, total_layers=total_mamba_blocks)
        
        self.w_mamba = mamba_block_cls(d_model=embed_dim, n_layer=n_mamba_per_block, total_layers=total_mamba_blocks)
        self.r_w_mamba = mamba_block_cls(d_model=embed_dim, n_layer=n_mamba_per_block, total_layers=total_mamba_blocks)
        
        assert forward_type == 'serial', f'forward_type should be "serial", but got {forward_type}'
        
    def forward(self, x: torch.Tensor, paired_reg_tokens_mask):  # [N, L, H, W, D]
        N, L, H, W, D = x.shape
        hw_mask, lw_mask, lh_mask = paired_reg_tokens_mask
        
        x = x.permute(0, 2, 3, 1, 4)  # [N, H, W, L, D]
        temp = x[hw_mask].reshape(-1, L, D)  # [N * HW_reg, L, D]
        temp = self.l_mamba(temp)
        temp = self.r_l_mamba(temp.flip(1)).flip(1)
        x[hw_mask] = temp.flatten(0, 1)
        
        x = x.permute(0, 3, 2, 1, 4)  # [N, L, W, H, D]
        temp = x[lw_mask].reshape(-1, H, D)  # [N * LW_reg, H, D]
        temp = self.h_mamba(temp)
        temp = self.r_h_mamba(temp.flip(1)).flip(1)
        x[lw_mask] = temp.flatten(0, 1)
        
        x = x.permute(0, 1, 3, 2, 4)  # [N, L, H, W, D]
        temp = x[lh_mask].reshape(-1, W, D)  # [N * LH_reg, W, D]
        temp = self.w_mamba(temp)
        temp = self.r_w_mamba(temp.flip(1)).flip(1)
        x[lh_mask] = temp.flatten(0, 1)
        
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
    
    
class VideoMamba3DExtra(nn.Module):
    def __init__(self, one_block_sample: Mamba3D, extra_blocks_on_global: int, mamba_cls: Mamba, n_mamba_per_block, mamba_3d_forward_type):
        super().__init__()
        
        self.extra_blocks_on_global = extra_blocks_on_global
        if self.extra_blocks_on_global > 0:
            mamba_layer_per_macro = 6
            self.extra_blocks = nn.ModuleList([
                Mamba3DExtra(mamba_cls, one_block_sample.embed_dim, n_mamba_per_block, total_mamba_blocks=self.extra_blocks_on_global * n_mamba_per_block * mamba_layer_per_macro, forward_type=mamba_3d_forward_type)
                for _ in range(self.extra_blocks_on_global)
                ])
        else:
            self.extra_blocks = nn.Identity()
    
    def forward(self, x: torch.Tensor, l_reg_token_indices: List[int], h_reg_token_indices: List[int], w_reg_token_indices: List[int]):  # [N, LLL, HHH, WWW, D]
        if self.extra_blocks_on_global > 0:
            paired_reg_tokens_mask = get_paired_reg_tokens_mask(x, l_reg_token_indices, h_reg_token_indices, w_reg_token_indices)
            for extra_block in self.extra_blocks:
                x = extra_block(x, paired_reg_tokens_mask)
        else:
            x = self.extra_blocks(x)
        return x
    
    
class VideoMamba3DFinal(nn.Module):
    def __init__(self, cfg, one_block_sample: Mamba3D):
        super().__init__()
        
        self.final_norm = nn.LayerNorm(one_block_sample.embed_dim)
        
        self.final_dim = getattr(cfg.model, 'final_dim', one_block_sample.embed_dim)
        self.final_aggregate_mode = getattr(cfg.model, 'final_aggregate_mode', 'all_mean')
        
        if self.final_aggregate_mode in ['all_mean', 'all_max']:
            self.final_linear = MLP(one_block_sample.embed_dim, [self.final_dim], nn.SiLU, final_activation=True)
        elif self.final_aggregate_mode == 'all_concat':
            if getattr(cfg.model, 'unified_reg_num', None) is not None:
                l_reg_num = h_reg_num = w_reg_num = cfg.model.unified_reg_num
            else:
                l_reg_num = getattr(cfg.model, 'l_reg_num', 0)
                h_reg_num = getattr(cfg.model, 'h_reg_num', 0)
                w_reg_num = getattr(cfg.model, 'w_reg_num', 0)
            num_used_global = (2 + l_reg_num) * (2 + h_reg_num) * (2 + w_reg_num)
            self.final_linear = MLP(num_used_global * one_block_sample.embed_dim, [self.final_dim], nn.SiLU, final_activation=True)
        elif self.final_aggregate_mode == 'all_enclosure_mean':
            self.final_linear = MLP(one_block_sample.embed_dim, [self.final_dim], nn.SiLU, final_activation=True)
        elif self.final_aggregate_mode == 'two_corner_concat':
            self.final_linear = MLP(2 * one_block_sample.embed_dim, [self.final_dim], nn.SiLU, final_activation=True)
        else:
            raise ValueError(f'final_aggregate_mode {self.final_aggregate_mode} not supported')
        
    def forward(self, x: torch.Tensor, l_reg_token_indices: List[int], h_reg_token_indices: List[int], w_reg_token_indices: List[int]):  # [N, LLL, HHH, WWW, D]
        if self.final_aggregate_mode == 'all_enclosure_mean':
            global_indices = torch.zeros_like(x[:, :, :, :, 0], dtype=torch.bool)  # [N, LLL, HHH, WWW]
            global_indices[:, l_reg_token_indices, :, :] = True
            global_indices[:, :, h_reg_token_indices, :] = True
            global_indices[:, :, :, w_reg_token_indices] = True
            x = x[global_indices].reshape(x.shape[0], -1, x.shape[-1])  # [N, n_enclosure, D]
            x = self.final_norm(x)
            x = x.mean(dim=1)  # [N, D]
            x = self.final_linear(x)
            return x
        elif self.final_aggregate_mode == 'two_corner_concat':
            x1 = x[:, 0, 0, 0, :]  # [N, D]
            x2 = x[:, -1, -1, -1, :]  # [N, D]
            x = torch.stack([x1, x2], dim=1)  # [N, 2, D]
            x = self.final_norm(x)
            x = x.flatten(1)  # [N, 2 * D]
            x = self.final_linear(x)
            return x
            
        x = x[:, l_reg_token_indices][:, :, h_reg_token_indices][:, :, :, w_reg_token_indices].flatten(1, -2)  # [N, n_global, D]
        x = self.final_norm(x)
        
        if self.final_aggregate_mode == 'all_mean':
            x = x.mean(dim=1)  # [N, D]
        elif self.final_aggregate_mode == 'all_max':
            x = x.max(dim=1).values  # [N, D]
        elif self.final_aggregate_mode == 'all_concat':
            x = x.flatten(1)  # [N, n_global * D]
        
        x = self.final_linear(x)
        return x


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


def get_paired_reg_tokens_mask(x: torch.Tensor, l_reg_token_indices: List[int], h_reg_token_indices: List[int], w_reg_token_indices: List[int]):
    _, LL, HH, WW, _ = x.shape
    
    l_reg_tokens_mask = torch.zeros(LL, dtype=torch.bool, device=x.device)
    l_reg_tokens_mask[l_reg_token_indices] = True
    
    h_reg_tokens_mask = torch.zeros(HH, dtype=torch.bool, device=x.device)
    h_reg_tokens_mask[h_reg_token_indices] = True
    
    w_reg_tokens_mask = torch.zeros(WW, dtype=torch.bool, device=x.device)
    w_reg_tokens_mask[w_reg_token_indices] = True
    
    hw_mask = torch.zeros(*x.shape[:-1], dtype=torch.bool, device=x.device)
    hw_mask[:, :, h_reg_tokens_mask, :] = True
    hw_mask[:, :, :, w_reg_tokens_mask] = True
    hw_mask = hw_mask.permute(0, 2, 3, 1)  # for [N, H, W, L]
    
    lw_mask = torch.zeros(*x.shape[:-1], dtype=torch.bool, device=x.device)
    lw_mask[:, l_reg_tokens_mask, :, :] = True
    lw_mask[:, :, :, w_reg_tokens_mask] = True
    lw_mask = lw_mask.permute(0, 1, 3, 2)  # for [N, L, W, H]
    
    lh_mask = torch.zeros(*x.shape[:-1], dtype=torch.bool, device=x.device)
    lh_mask[:, l_reg_tokens_mask, :, :] = True
    lh_mask[:, :, h_reg_tokens_mask, :] = True
    # lh_mask = lh_mask.permute(0, 1, 2, 3)  # for [N, L, H, W]
    
    return hw_mask, lw_mask, lh_mask