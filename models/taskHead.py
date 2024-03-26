import math
from typing import List

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# cls
class LinearClsHead(nn.Module):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 mode: str
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.mode = mode

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, x):
        if isinstance(x, List):
            hidden_states = torch.zeros_like(x[0])
            for item in x:
                hidden_states.add_(item)
        else:
            hidden_states = x
        
        if self.mode == "cls_token":
            hidden_states = hidden_states[:, 0]
        elif self.mode == "mean":
            hidden_states = hidden_states.mean(1)
            
        cls_score = self.fc(hidden_states)
        return cls_score
    
    def loss(self, cls_score, target) -> torch.Tensor:
        loss = self.loss_fn(cls_score, target)
        return loss


# Seg
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
    
class DenseTransform(nn.Module):
    def __init__(self, embed_dim, out_dim_list=[128, 256, 512, 512]):
        super().__init__()
        
        self.upsampleX4 = nn.Sequential(
            UpSampleX2Module(embed_dim, embed_dim), 
            UpSampleX2Module(embed_dim, out_dim_list[0])
        )
        
        self.upsampleX2 = UpSampleX2Module(embed_dim, out_dim_list[1])
        
        self.upsampleX1 = nn.Conv2d(embed_dim, out_dim_list[2], kernel_size=3, padding=1)
        
        self.downsample2 = nn.Conv2d(embed_dim, out_dim_list[3], kernel_size=2, stride=2)

    def forward(self, feats):
        assert len(feats) == 4
        h = int(math.sqrt(feats[0].shape[1]))
        assert h**2 == feats[0].shape[1]
        feats = [einops.rearrange(item, "B (h w) D -> B D h w", h=h) for item in feats]
        
        feats[0] = self.upsampleX4(feats[0])
        feats[1] = self.upsampleX2(feats[1])
        feats[2] = self.upsampleX1(feats[2])
        feats[3] = self.downsample2(feats[3])
        
        return feats


class UpSampleX2Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.module1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.GELU()
        )
        
        self.module2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        
    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        return x
    

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes, in_channels=[32, 64, 160, 256], embedding_dim=256, 
                 dropout_ratio=0.1, ignore_index=-100):
        super().__init__()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Conv2d(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        
        self.dropout = nn.Dropout2d(dropout_ratio)
        
        self.ignore_index = ignore_index

    def forward(self, x):
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
    
    def loss(self, seg_logit, seg_label):
        seg_logit = F.interpolate(
            input=seg_logit,
            size=seg_label.shape[1:],
            mode='bilinear',
            align_corners=False)
        seg_label = seg_label.squeeze(1)
        loss = F.cross_entropy(
            seg_logit,
            seg_label,
            reduction='mean',
            ignore_index=self.ignore_index)
        return loss