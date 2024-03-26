import math
import os
import einops

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
from pathlib import Path


class TokenModel(nn.Module):
    def __init__(self, tokenizer, backbone, taskHead):
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.taskHead = taskHead
        self.imgs = []
        self.attns = []
        
        self.attn = None
        self.attn_head = None
        
    def forward(self, x):
        x, self.attn = self.tokenizer(x)
        x = self.backbone(x)
        return self.taskHead(x)
    
    def save_attn(self, x):
        x = x.detach().cpu()
        img = (x - torch.min(x))/(torch.max(x) - torch.min(x))
        img = img.permute(0, 2, 3, 1)
            
        self.imgs.append(img)
        
        self.attn_head = self.attn.shape[1]
        self.attn = einops.rearrange(self.attn, "b h n N -> (b h) n N")
        
        max_v = torch.max(self.attn, dim=1, keepdim=True)[0]
        min_v = torch.min(self.attn, dim=1, keepdim=True)[0]
        self.attn = (self.attn - min_v) / (max_v - min_v)

        attn_vis = F.softmax(self.attn, dim=1)
        attn_vis = torch.argmax(attn_vis, dim=1).float().detach().cpu()
        h = int(math.sqrt(attn_vis.shape[-1]))
        attn_vis = F.interpolate(attn_vis.reshape((-1, h, h)).unsqueeze(1), 
                                size=x.shape[-2:], mode="nearest").squeeze()
        attn_vis = (attn_vis - torch.min(attn_vis))/(torch.max(attn_vis) - torch.min(attn_vis))
        
        self.attns.append(attn_vis)
            
    def save_attn_to_file(self, save_path, cur_epoch):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        if cur_epoch == 0:
            self.imgs = torch.concatenate(self.imgs, dim=0)
        self.attns = torch.concatenate(self.attns, dim=0)

        for i in tqdm(range(self.attns.shape[0])):
            cur_save_path = os.path.join(save_path, f"id{i}.jpg")
            if cur_epoch == 0:
                save_image(img=self.imgs[i], attn_vis=self.attns[i*self.attn_head:(i+1)*self.attn_head], save_path=cur_save_path)
            else:
                save_image(img=None, attn_vis=self.attns[i*self.attn_head:(i+1)*self.attn_head], save_path=cur_save_path)
        
        self.imgs = []
        self.attns = []


def save_image(img, attn_vis, save_path):
    if img != None:
        img_show = [(img.numpy() * 255).astype('uint8')]
    else:
        img_show = []
    
    for attn in attn_vis:
        attn_vis_np = (attn.numpy() * 255).astype('uint8')
        attn_vis_colored = cv2.applyColorMap(attn_vis_np, cv2.COLORMAP_RAINBOW)
        img_show.append(attn_vis_colored)
    
    combined_img = cv2.hconcat(img_show)
    cv2.imwrite(save_path, combined_img)
        
            
        
        