import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class S2DAdapter(nn.Module):
    def __init__(self, scale_size):
        super().__init__()
        
        self.scale_size = scale_size
     
    def forward(self, x, attn_map):
        # x: B n D  map: B n N
        edge = int(math.sqrt(attn_map.size(-1)))
        attn_map = einops.rearrange(attn_map, "B n (h w) -> B n h w", h=edge, w=edge)
        attn_map_resize = F.interpolate(attn_map, (self.scale_size, self.scale_size))
        attn_map_resize = einops.rearrange(attn_map_resize, "B n h w -> B (h w) n") 
        # scale to [0, 1] for softmax (too small!)
        max_v = torch.max(attn_map_resize, dim=-1, keepdim=True)[0]
        min_v = torch.min(attn_map_resize, dim=-1, keepdim=True)[0]
        attn_map_resize = (attn_map_resize - min_v) / (max_v - min_v)
        
        attn_map_resize = F.softmax(attn_map_resize, dim=-1) 
        recon_x = attn_map_resize.detach() @ x
        
        return recon_x
    

# For baseline
class LinearAdapter(nn.Module):
    def __init__(self, hidden_num, out_num, with_cls_token=True):
        super().__init__()
        self.with_cls_token = with_cls_token
        
        self.module = nn.Sequential(
            nn.Linear(hidden_num, out_num),
            nn.GELU(),
            nn.Linear(out_num, out_num),
        )

    def forward(self, x):
        
        if self.with_cls_token:
            x = x[:, 1:, :]
        
        x = einops.rearrange(x, "B N D -> B D N")
        recon_x = self.module(x)
        recon_x = einops.rearrange(recon_x, "B D N -> B N D")
        
        return recon_x


# if __name__ == "__main__":
#     x = torch.randn((64, 4, 512))
#     map_ = torch.randn((64, 4, 196))
#     model = S2DAdapter(512)
#     out = model(x, map_)
#     print(out.shape)
    