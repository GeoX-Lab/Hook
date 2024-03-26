from functools import partial
from typing import Callable, Optional, Tuple

import einops
import einops_exts
import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import to_2tuple
from timm.models.vision_transformer_sam import (DropPath, LayerScale,
                                                window_partition,
                                                window_unpartition)


class VisTokenizer(nn.Module):
    def __init__(self, img_size=224, in_channels=3, downsample_dim=512, token_num=4, 
                 out_dim=768, with_cls_token=True):
        super().__init__()
        self.downsample_dim = downsample_dim
        self.downsample_feat_h = img_size // 4
        self.with_cls_token = with_cls_token
        
        self.downsample_module = DownsampleConv(in_channels=in_channels, out_dim=self.downsample_dim)
        
        self.semantic_module = nn.Sequential(*[
            # Local Window Attn
            SAM_Block(
                dim=self.downsample_dim,
                out_dim=self.downsample_dim,
                num_heads=8,
                window_size=self.downsample_feat_h // 2
            ),
            # Global Attn
            SAM_Block(
                dim=self.downsample_dim,
                out_dim=self.downsample_dim,
                num_heads=8,
                window_size=0
            ),
        ])
        
        # self.semantic_module = nn.Identity()
        
        self.pos_embed = nn.Parameter(torch.randn(1, self.downsample_feat_h ** 2, self.downsample_dim) * .02)
        
        self.query = nn.Parameter(torch.randn(token_num, out_dim))
        self.match_attn = PerceiverAttention(self.downsample_dim, out_dim=out_dim, dim_head=self.downsample_dim // 2, heads=1)
        self.match_ff = FeedForward(dim=out_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, out_dim)) if self.with_cls_token else None
        
    def forward(self, x):
        # downsample
        downsample_feat = self.downsample_module(x)
        B, D, H, W = downsample_feat.shape
        downsample_feat = einops.rearrange(downsample_feat, "B d H W -> B (H W) d")
        downsample_feat = downsample_feat + self.pos_embed
        
        # extract semantic 
        downsample_feat = einops.rearrange(downsample_feat, "B (H W) d -> B H W d", H=H, W=W)
        semantic_feat = self.semantic_module(downsample_feat)
        semantic_feat = einops.rearrange(semantic_feat, "B H W d -> B (H W) d")
        
        # match token
        latents = einops.repeat(self.query, 'n d -> b n d', b=semantic_feat.shape[0])
        tokens, attn_map = self.match_attn(semantic_feat, latents)
        tokens = tokens + latents
        tokens = self.match_ff(tokens) + tokens
            
        if self.cls_token is not None:
            tokens = torch.cat((self.cls_token.expand(tokens.shape[0], -1, -1), tokens), dim=1)
        
        return tokens, attn_map


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

class SAM_Block(nn.Module):

    def __init__(
            self,
            dim,
            out_dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_norm=False,
            proj_drop=0.,
            attn_drop=0.,
            init_values=None,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            mlp_layer=Mlp,
            window_size=0
    ):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
            out_features=out_dim,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
                    
    def forward(self, x):
        B, H, W, _ = x.shape

        shortcut = x
        x = self.norm1(x)
        # Window partition
        pad_hw: Optional[Tuple[int, int]] = None
        if self.window_size > 0:
            x, pad_hw = window_partition(x, self.window_size)

        x = self.drop_path1(self.ls1(self.attn(x)))

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, (H, W), pad_hw)

        x = shortcut + x

        x = x.reshape(B, H * W, -1)  # MLP is faster for N, L, C tensor
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = x.reshape(B, H, W, -1)

        return x


class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, _ = x.shape
        N = H * W
        x = x.reshape(B, N, -1)
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # qkv with shape (3, B, nHead, H * W, C)
        q, k, v = qkv.unbind(0)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k = self.q_norm(q), self.k_norm(k)
        
        x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )

        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = x.view(B, H, W, -1)
        return x

    
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False)
    )


class PerceiverAttention(nn.Module):
    def __init__(
            self,
            input_dim,
            out_dim,
            dim_head=64,
            heads=8
    ):
        super().__init__()

        self.input_dim = input_dim

        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(input_dim)
        self.norm_latents = nn.LayerNorm(out_dim)

        self.to_q = nn.Linear(out_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(input_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, out_dim, bias=False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        kv_input = x
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = einops_exts.rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=h)

        q = q * self.scale

        # attention
        sim = torch.einsum('... i d, ... j d  -> ... i j', q, k)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()        
        attn = sim.softmax(dim=-1)

        out = torch.einsum('... i j, ... j d -> ... i d', attn, v)
        out = einops.rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out), attn


class DownsampleConv(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1, padding=0)
        
        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU()
        )
        
        self.module2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        
        self.module3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU()
        )
        
        self.module4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=out_dim, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv1(x)
        x = x + self.module1(x)
        x = self.module2(x)
        x = x + self.module3(x)
        x = self.module4(x)
        x = self.conv2(x)
        return x
    
        
class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, embed_dim=768, with_cls_token=True):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if with_cls_token else None
        
    def forward(self, inputs):
        x = self.proj(inputs)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
        return x

