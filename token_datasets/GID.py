import os

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


class GIDDataset(Dataset):
    def __init__(self, img_dir, ann_dir, is_train=True):
        self.imgs = []
        self.labels = []
        
        self.ignore_idx = 255
        self.num_classes = 5
        
        img_list = os.listdir(img_dir)
        print("load image ......")
        for img in tqdm(img_list):
            self.imgs.append(Image.fromarray(np.array(Image.open(os.path.join(img_dir, img)))))
            label = torch.tensor(np.array(Image.open(os.path.join(ann_dir, img))))
            label[label == 0] = self.ignore_idx
            label = label - 1
            label[label == (self.ignore_idx-1)] = self.ignore_idx
            self.labels.append(label)
            
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.RandomApply(
                    torch.nn.ModuleList([
                        transforms.GaussianBlur(kernel_size=(5, 9))
                    ]),
                    p=0.5
                ),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.CLASSES = ("Built-up", "Framland", "Forest", "Meadow", "Water")
        
        # 5 classes + other        
        self.PALETTE = [
            [0, 0, 255],
            [0, 255, 255],
            [0, 255, 0],
            [255, 255, 0],
            [102, 137, 249],
            [255, 255, 255]
        ]
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    @staticmethod
    def get_cfg():
        return dict(
            train_imgdir = "/media/shaorun/zl/Dataset/HOOK_datasets/GID/train",
            train_anndir = "/media/shaorun/zl/Dataset/HOOK_datasets/GID/train_labels",
            test_imgdir = "/media/shaorun/zl/Dataset/HOOK_datasets/GID/test",
            test_anndir = "/media/shaorun/zl/Dataset/HOOK_datasets/GID/test_labels",
        )