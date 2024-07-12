import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from typing import List, Tuple, Union, Dict, Callable
import warnings
import copy
from smoothllm import GradModded


def GradmodGPTNeoAttn(model, kv_cache_grad_buf):
    # warnings.warn("I have absolutely no idea whether it works on other models. Take care")

    for module, accumulated_kv_grad in zip(model.transformer.h, kv_cache_grad_buf):
        # make a hook injecting gradients from target_kv_grad_cache into the attention operators
        attn_module = module.attn.attention
        hd = attn_module.head_dim
        nh = attn_module.num_heads

        k_hook = lambda _, nh=nh, hd=hd: attn_module._merge_heads(
            accumulated_kv_grad[0], nh, hd
        )

        module.attn.attention.k_proj = GradModded(module.attn.attention.k_proj, k_hook)

        v_hook = lambda _, nh=nh, hd=hd: attn_module._merge_heads(
            accumulated_kv_grad[1], nh, hd
        )
        module.attn.attention.v_proj = GradModded(module.attn.attention.v_proj, v_hook)

    return model


def UngradmodGPTNeoAttn(model):
    for module in model.transformer.h:
        module.attn.attention.k_proj = module.attn.attention.k_proj.base_layer
        module.attn.attention.v_proj = module.attn.attention.v_proj.base_layer
