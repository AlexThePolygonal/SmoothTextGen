import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from typing import List, Tuple, Union, Dict, Callable
import warnings
import copy

class GradMod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, hook):
        ctx.hook = hook
        return input

    @torch.autograd.function.once_differentiable
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output + ctx.hook(grad_output), None

class GradModded(nn.Module):
    base_layer : nn.Module
    gradmod : GradMod
    hook : Callable

    def __init__(self, base : nn.Module, hook : Callable):
        super().__init__()
        self.base_layer = base
        self.gradmod = GradMod()
        self.hook = hook
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        return self.gradmod.apply(self.base_layer(input), self.hook)

# TODO fix this crap
# TODO remove debugging and attach proper hooks
def GradmodGPTNeoAttn(model, target_kv_cache):
    warnings.warn("I have absolutely no idea whether it works on other models. Take care")
    
    
    for module, kv in zip(model.transformer.h, target_kv_cache):
        attn_module = module.attn.attention
        hd = attn_module.head_dim
        nh = attn_module.num_heads

        # k_hook = lambda x, nh=nh, hd=hd : print(attn_module._split_heads(x,nh,hd).shape)
        k_hook = lambda _, nh=nh, hd=hd :\
            attn_module._merge_heads(kv[0],nh,hd)
        
        module.attn.attention.k_proj = GradModded(module.attn.attention.k_proj, k_hook)
        
        v_hook = lambda _, nh=nh, hd=hd :\
            attn_module._merge_heads(kv[1],nh,hd)
        module.attn.attention.v_proj = GradModded(module.attn.attention.v_proj, v_hook)
    
    return model

def UngradmodGPTNeoAttn(model):
    for module in model.transformer.h:
        module.attn.attention.k_proj = module.attn.attention.k_proj.base_layer
        module.attn.attention.v_proj = module.attn.attention.v_proj.base_layer