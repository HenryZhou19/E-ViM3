from functools import partial
from typing import Optional, Union

import torch
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, rms_norm_fn
from timm.models.layers import DropPath
from torch import nn

from src.utils.misc import DummyContextManager

from .modules.bimamba2_block import BiMamba2Block
from .modules.bimamba_block import BiMambaBlock
from .modules.mamba2_block import Mamba2Block
from .modules.mamba_block import MambaBlock
from .modules.norm import RMSNorm
from .modules.utils import init_mamba_weights


class MambaLayer(nn.Module):
    def __init__(
        self,
        layer_index : int,
        mamba_block_cls: MambaBlock,
        norm_cls: Union[nn.LayerNorm, RMSNorm],
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
        ):
        '''
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"
        
        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        '''
        super().__init__()
        self.layer_index = layer_index
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        
        self.mamba_block: MambaBlock = mamba_block_cls(layer_index=layer_index)
        self.norm = norm_cls()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        inference_params=None,
        ):
        r"""Pass the input through the encoder layer.
        
        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = mamba_block(norm(residual))
        """
        
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )
        
        hidden_states = self.mamba_block(hidden_states, inference_params=inference_params)
        return hidden_states, residual
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mamba_block.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class Mamba(nn.Module):
    def __init__(
        self,
        n_layer: int,
        d_model: int,
        mamba_block_cls: Union[MambaBlock, BiMambaBlock, Mamba2Block, BiMamba2Block],
        mamba_block_config: dict,
        rms_norm=False,
        norm_epsilon=1e-5,
        final_norm=True,
        residual_in_fp32: bool=False,
        no_amp=True,
        custom_init=True,
        initializer_cfg=None,
        fused_add_norm=False,
        total_layers: int=None,
        drop_path=0.0,
        device=None,
        dtype=None,
        ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.no_amp = no_amp
        
        norm_cls = partial(
            nn.LayerNorm if not rms_norm else RMSNorm,
            normalized_shape=d_model,
            eps=norm_epsilon,
            **factory_kwargs,
            )
        mamba_block_cls = partial(
            mamba_block_cls,
            d_model=d_model,
            **mamba_block_config,
            **factory_kwargs,
            )
        
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        
        self.mamba_layers = nn.ModuleList(
            [
                MambaLayer(
                    layer_index=layer_index,
                    mamba_block_cls=mamba_block_cls,
                    norm_cls=norm_cls,
                    fused_add_norm=fused_add_norm,
                    residual_in_fp32=residual_in_fp32,
                    drop_path=drop_path,
                )
                for layer_index in range(n_layer)
            ]
        )
        
        if final_norm:
            self.final_norm: Union[nn.LayerNorm, RMSNorm] = norm_cls()
        else:
            self.final_norm = None
        
        self.total_layers = n_layer if total_layers is None else total_layers
        self.initializer_cfg = initializer_cfg
        if custom_init:
            self._init_weights()
            
    def _init_weights(self):
        self.apply(
            partial(
                init_mamba_weights,
                n_layer=self.total_layers,
                **(self.initializer_cfg if self.initializer_cfg is not None else {}),
            )
        )
            
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            idx: mamba_layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for idx, mamba_layer in enumerate(self.mamba_layers)
        }
        
    def forward(self, hidden_states: torch.Tensor, inference_params=None) -> torch.Tensor:
        disable_amp = torch.is_autocast_enabled() and self.no_amp
        if disable_amp:
            hidden_states = hidden_states.float()
            context = partial(torch.cuda.amp.autocast, enabled=False)
        else:
            context = DummyContextManager
        
        with context():
            residual = None
            for layer in self.mamba_layers:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
            
            if self.final_norm is not None:
                if not self.fused_add_norm:
                    residual = (hidden_states + residual) if residual is not None else hidden_states
                    hidden_states = self.final_norm(residual.to(dtype=self.final_norm.weight.dtype))
                else:
                    # Set prenorm=False here since we don't need the residual
                    hidden_states = layer_norm_fn(
                        hidden_states,
                        self.final_norm.weight,
                        self.final_norm.bias,
                        eps=self.final_norm.eps,
                        residual=residual,
                        prenorm=False,
                        residual_in_fp32=self.residual_in_fp32,
                        is_rms_norm=isinstance(self.final_norm, RMSNorm)
                    )
            else:
                hidden_states = (hidden_states + residual) if residual is not None else hidden_states
        
        return hidden_states
