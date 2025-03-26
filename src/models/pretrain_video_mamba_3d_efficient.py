import math
from copy import deepcopy

import torch
from torch import nn

from .modules.model_base import ModelBase, model_register


@model_register('pretrain_video_mamba_3d_efficient')  # decoder uses the unmasked patches of encoder's outputs to predict the masked patches
class PretrainVideoMamba3DEfficientModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        assert getattr(cfg.model, 'new_scan', False)
        from .modules.video_mamba_3d_new_efficient import (
            VideoMamba3D, VideoMamba3DDecoderForPretraining)
            
        self.net = VideoMamba3D(cfg)
        
        self.pretrain_decoder_l_pos_embed = deepcopy(self.net.l_pos_embed)
        self.pretrain_decoder_h_pos_embed = deepcopy(self.net.h_pos_embed)
        self.pretrain_decoder_w_pos_embed = deepcopy(self.net.w_pos_embed)
        
        # assert cfg.model.decoder_additional_blocks % 2 == 0, 'decoder_additional_blocks should be divided by 2'
        decoder_additional_blocks = cfg.model.decoder_additional_blocks
        self.pretrain_decoder = VideoMamba3DDecoderForPretraining(
            additional_blocks=decoder_additional_blocks,
            mamba_cls=self.net.mamba_cls,
            n_mamba_per_block=cfg.model.n_mamba_per_block,
            hidden_dim=self.net.embed_dim,
            out_channels=self.net.in_channels,
            time_patch_size=self.net.time_patch_size,
            patch_size=self.net.patch_size,
            mamba_3d_forward_type=getattr(cfg.model, 'mamba_3d_forward_type', 'serial'),
            )
        
        self.encoder_random_patch_mask_maker = RandomPatchMaskMaker(
            self.net.embed_dim,
            cfg.model.mask_prob,
            decoder_all_mask=getattr(cfg.model, 'decoder_all_mask', False),
            time_mask_chain=getattr(cfg.model, 'time_mask_chain', 1),
            mask_chain=getattr(cfg.model, 'mask_chain', 1),
            )

        self.decoder_random_patch_mask_maker = RandomPatchMaskMaker(
            self.net.embed_dim,
            None,
            )
    
    def forward(self, inputs: dict) -> dict:
        x: torch.Tensor = inputs['x']  # [N, C, L_all, H, W]
        
        x, x_original = self._do_external_patch_embed(x)  # [N, LL, HH, WW, D], [N, LL, HH, WW, C*time_patch_size*patch_size*patch_size]
        
        # add mask token embed for encoder (discard latter)
        x, patch_mask, patch_mask_for_decoder = self.encoder_random_patch_mask_maker(x)  # [N, LL, HH, WW, D], [N, LL, HH, WW]

        x = self.net(x, external_patch_embed=True, patch_mask=patch_mask)  # [N, 1 + LL + 1, 1 + HH + 1, 1 + WW + 1, D]
        
        N, D = x.shape[0], x.shape[-1]
        
        # discard the masked patches' outputs of encoder, and add new mask token embed for decoder
        x_remasked, _, __ = self.decoder_random_patch_mask_maker(
            x[self.net.real_tokens_mask.expand(N, -1, -1, -1)].reshape(N, *self.net.patch_embed.grid_size, D),  # [N, LL, HH, WW, D]
            patch_mask_for_decoder,
            )
        x[self.net.real_tokens_mask.expand(N, -1, -1, -1)] = x_remasked.flatten(0, -2)
        
        # re-add the pos embed (shared between encoder and decoder)
        x = x + self.pretrain_decoder_l_pos_embed.expand(N, -1, -1, -1, -1)
        x = x + self.pretrain_decoder_h_pos_embed.expand(N, -1, -1, -1, -1)
        x = x + self.pretrain_decoder_w_pos_embed.expand(N, -1, -1, -1, -1)
        
        x = self.pretrain_decoder(x, real_tokens_mask=self.net.real_tokens_mask.expand(N, -1, -1, -1), grid_size=self.net.patch_embed.grid_size)  # [N, LL, HH, WW, C*time_patch_size*patch_size*patch_size]
        
        x_original_masked, x_pred_masked = self._get_original_and_pred_patches(x_original, x, patch_mask_for_decoder)
        
        return {
            'original_masked': x_original_masked,  # [N_masked, patch_channels_all=C*time_patch_size*patch_size*patch_size]
            'pred_masked': x_pred_masked,  # [N_masked, patch_channels_all=C*time_patch_size*patch_size*patch_size]
            'patch_original': x_original,  # [N, LL, HH, WW, C*time_patch_size*patch_size*patch_size]
            'patch_mask': patch_mask_for_decoder,  # [N, LL, HH, WW], 'True' for masked, 'False' for not masked
        }

    def _do_external_patch_embed(self, x: torch.Tensor) -> torch.Tensor:  # [N, C, L_all, H, W]
        time_patch_size = self.net.time_patch_size
        patch_size = self.net.patch_size
        x_original = x.unfold(2, time_patch_size, time_patch_size).unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)  # [N, C, LL, HH, WW, time_patch_size, patch_size, patch_size]
        x_original = x_original.permute(0, 2, 3, 4, 1, 5, 6, 7)  # [N, LL, HH, WW, C, time_patch_size, patch_size, patch_size]
        x_original = x_original.flatten(-4)  # [N, LL, HH, WW, C*time_patch_size*patch_size*patch_size]
        
        x = self.net.patch_embed(x)  # [N, LL, HH, WW, D]
        return x, x_original
    
    def _get_original_and_pred_patches(self, x_original, x_pred, patch_mask):
        '''
        x_original: [N, LL, HH, WW, C*time_patch_size*patch_size*patch_size]
        x_pred: [N, LL, HH, WW, C*time_patch_size*patch_size*patch_size]
        patch_mask: bool [N, LL, HH, WW], 'True' for masked, 'False' for not masked
        '''
        x_original_masked = x_original[patch_mask]
        x_pred_masked = x_pred[patch_mask]
        return x_original_masked, x_pred_masked
        

class RandomPatchMaskMaker(nn.Module):
    def __init__(self, embed_dim: int, mask_prob: float=None, decoder_all_mask=False, time_mask_chain:int=1, mask_chain:int=1):
        super().__init__()
        self.mask_prob = mask_prob
        self.decoder_all_mask = decoder_all_mask
        
        self.time_mask_chain = time_mask_chain
        self.mask_chain = mask_chain
        if self.time_mask_chain > 1 or self.mask_chain > 1:
            self.do_chain_mask = True
        else:
            self.do_chain_mask = False
        
        self.mask_token = nn.Parameter(torch.randn(1, embed_dim))
        
    @staticmethod
    def random_bool_vector(N, L, K, device):
        bool_vectors = torch.zeros((N, L), dtype=torch.bool, device=device)
        indices = torch.multinomial(torch.ones(N, L, device=device), K, replacement=False)
        bool_vectors[torch.arange(N).unsqueeze(1), indices] = True
        return bool_vectors
    
    def forward(self, x: torch.Tensor, patch_mask=None) -> torch.Tensor:  # [N, LL, HH, WW, D]
        N, LL, HH, WW, D = x.shape
        
        if patch_mask is None:  # encoder
            assert self.mask_prob is not None
            if self.do_chain_mask:
                LLL = math.ceil(LL / self.time_mask_chain)
                HHH = math.ceil(HH / self.mask_chain)
                WWW = math.ceil(WW / self.mask_chain)
                patch_length = LLL * HHH * WWW
                patch_mask_length = int(self.mask_prob * patch_length)
                if patch_mask_length == 0:
                    patch_mask = torch.zeros(N, LLL, HHH, WWW, device=x.device).bool()
                else:
                    patch_mask = self.random_bool_vector(1, patch_length, patch_mask_length, x.device).reshape(1, LLL, HHH, WWW)  # [1, LL, HH, WW]
                patch_mask = torch.nn.functional.interpolate(patch_mask.float().unsqueeze(1), size=(LL, HH, WW), mode='nearest').squeeze(1).bool()  # [1, LL, HH, WW]
                patch_mask = patch_mask.expand(N, -1, -1, -1)  # [N, LL, HH, WW]
            else:
                patch_length = LL * HH * WW
                patch_mask_length = int(self.mask_prob * patch_length)
                if patch_mask_length == 0:
                    patch_mask = torch.zeros(N, LL, HH, WW, device=x.device).bool()
                else:
                    patch_mask = self.random_bool_vector(N, patch_length, patch_mask_length, x.device).reshape(N, LL, HH, WW)
        
            x[patch_mask] = self.mask_token.to(x.dtype)  # [N, LL, HH, WW, D]
        
            if self.decoder_all_mask:
                patch_mask_for_decoder = torch.ones(N, LL, HH, WW, device=x.device).bool()
            else:
                patch_mask_for_decoder = patch_mask
            if patch_mask_length == 0:
                patch_mask = None
                
        else:  # decoder
            x[patch_mask] = self.mask_token.to(x.dtype)  # [N, LL, HH, WW, D]
            patch_mask_for_decoder = None
        
        return x, patch_mask, patch_mask_for_decoder