import math
import os

import einops
import matplotlib

matplotlib.use('Agg')
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class TokenModel_Forseg(nn.Module):
    def __init__(self, tokenizer, backbone, s2dAdapter, denseTransform, taskHead):
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.s2dAdapter = s2dAdapter
        self.denseTransform = denseTransform
        self.taskHead = taskHead
        self.imgs = []
        self.attns = []
        
        self.attn = None
        self.attn_head = None
        
    def forward_feature(self, x):
        x, self.attn = self.tokenizer(x)
        xs = self.backbone(x)
        if self.s2dAdapter:
            xs = [self.s2dAdapter(item, self.attn.squeeze()) for item in xs]
        if self.denseTransform:
            xs = self.denseTransform(xs)
        pred = self.taskHead(xs)
        return pred
        
    def forward_train(self, x, label):
        pred = self.forward_feature(x)
        loss = self.taskHead.loss(pred, label)
        return loss
    
    def forward_test(self, x, label):
        pred = self.forward_feature(x)
        pred = F.interpolate(
            input=pred,
            size=label.shape[1:],
            mode='bilinear',
            align_corners=False)
        pred = F.softmax(pred, dim=1)
        pred = pred.argmax(dim=1)
        return pred
        
    def forward(self, x, label, is_train=True):
        if is_train:
            return self.forward_train(x, label)
        else:
            return self.forward_test(x, label)
    
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
        
    def save_segmap(self, imgs, labels, segs, PALETTE, save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        palette = np.array(PALETTE)
        labels[labels == 255] = len(PALETTE) - 1
        
        imgs = (imgs - torch.min(imgs))/(torch.max(imgs) - torch.min(imgs))
        imgs = imgs.permute(0, 2, 3, 1)
        
        for i in tqdm(range(imgs.shape[0])):
            cur_save_path = os.path.join(save_path, f"id{i}.jpg")
            
            img_show = [(imgs[i].numpy() * 255).astype('uint8')]
            label_color = labels[i].numpy()
            label_color = palette[label_color]
            img_show.append(label_color.astype('uint8'))
            
            seg_color = segs[i].numpy()
            seg_color = palette[seg_color]
            img_show.append(seg_color.astype('uint8'))
            
            combined_img = cv2.hconcat(img_show)
            cv2.imwrite(cur_save_path, combined_img)
            
        
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


class PeModel_Forseg(nn.Module):
    def __init__(self, tokenizer, backbone, denseTransform, taskHead, adapter=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.backbone = backbone
        self.denseTransform = denseTransform
        self.taskHead = taskHead
        self.adapter = adapter
        
    def forward_feature(self, x):
        x = self.tokenizer(x)
        xs = self.backbone(x)
        if self.adapter:
            xs = [self.adapter(item) for item in xs]
        if self.denseTransform:
            xs = self.denseTransform(xs)
        pred = self.taskHead(xs)
        return pred
        
    def forward_train(self, x, label):
        pred = self.forward_feature(x)
        loss = self.taskHead.loss(pred, label)
        return loss
    
    def forward_test(self, x, label):
        pred = self.forward_feature(x)
        pred = F.interpolate(
            input=pred,
            size=label.shape[1:],
            mode='bilinear',
            align_corners=False)
        pred = F.softmax(pred, dim=1)
        pred = pred.argmax(dim=1)
        return pred
        
    def forward(self, x, label, is_train=True):
        if is_train:
            return self.forward_train(x, label)
        else:
            return self.forward_test(x, label)
    
    def save_segmap(self, imgs, labels, segs, PALETTE, save_path):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        palette = np.array(PALETTE)
        labels[labels == 255] = len(PALETTE) - 1
        
        imgs = (imgs - torch.min(imgs))/(torch.max(imgs) - torch.min(imgs))
        imgs = imgs.permute(0, 2, 3, 1)
        
        for i in tqdm(range(imgs.shape[0])):
            cur_save_path = os.path.join(save_path, f"id{i}.jpg")
            
            img_show = [(imgs[i].numpy() * 255).astype('uint8')]
            label_color = labels[i].numpy()
            label_color = palette[label_color]
            img_show.append(label_color.astype('uint8'))
            
            seg_color = segs[i].numpy()
            seg_color = palette[seg_color]
            img_show.append(seg_color.astype('uint8'))
            
            combined_img = cv2.hconcat(img_show)
            cv2.imwrite(cur_save_path, combined_img)
    
        
            
        
        