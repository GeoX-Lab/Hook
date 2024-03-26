import os

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from utils import cls_accuracy
from utils.metrics import Avg_values


class WHURS19Dataset(Dataset):
    def __init__(self, metainfo_file, root_path, is_train=True, img_size=224):
        self.metainfo = pd.read_csv(metainfo_file)
        self.root_path = root_path
        self.target_transform = None
        
        if is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(img_size, interpolation=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.results = dict()
        
        # load images
        self.imgs = []
        for idx, row in tqdm(self.metainfo.iterrows()):
            img_path = os.path.join(self.root_path, row[1])
            self.imgs.append(Image.open(img_path).convert('RGB'))

    def __len__(self):
        return self.metainfo.shape[0]

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.metainfo.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def eval(self, cls_score, target, topk=(1, 5)):
        batch_size = cls_score.size(0)
        res = cls_accuracy(cls_score, target, topk)
        for key in res.keys():
            if key not in self.results.keys():
                self.results[key] = Avg_values()
            self.results[key].update(res[key].item(), batch_size)
    
    def get_eval_res(self):
        res = dict()
        for key in self.results.keys():
            res[key] = self.results[key].avg
        self.results = dict()
        return res
    
    @staticmethod
    def get_cfg():
        return dict(
            root_path='/media/shaorun/zl/Dataset/HOOK_datasets/RS19/WHU-RS19',
            metainfo_train='/media/shaorun/zl/Dataset/HOOK_datasets/RS19/train0.5.csv',
            metainfo_test='/media/shaorun/zl/Dataset/HOOK_datasets/RS19/test0.5.csv'
        )
