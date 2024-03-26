import torch.nn as nn 
from timm.models.vision_transformer import Block


class AttnBlocks(nn.Module):
    def __init__(self, depth=12, dim=768, head=12, ret_layers=[-1]):
        super().__init__()
        self.module = nn.Sequential(*[
            Block(
                dim=dim,
                num_heads=head,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(depth)])
        
        if len(ret_layers) == 1 and ret_layers[0] == -1:
            self.ret_layers = [depth-1]
        else:
            self.ret_layers = ret_layers
        
    def forward(self, x):
        rets = []
        for i, layer in enumerate(self.module):
            x = layer(x)
            if i in self.ret_layers:
                rets.append(x)
        return rets