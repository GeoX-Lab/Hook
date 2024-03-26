import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from utils import Avg_values, MY_Logger, adjust_learning_rate
from token_datasets import NWPURESIS45Dataset, WHURS19Dataset
from models import VisTokenizer, LinearClsHead, AttnBlocks, TokenModel


def get_args_parser():
    parser = argparse.ArgumentParser('Visual Tokenizer', add_help=False)
    
    # Common parameters
    parser.add_argument('--exp-name', default="debug", type=str, help='current experiment name')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--print-freq', default=20, type=int)
    
    # Model parameters
    parser.add_argument('--token-num', default=6, type=int)
    parser.add_argument('--vis', action="store_true")
    parser.add_argument('--downsample-dim', default=512, type=int)
    parser.add_argument('--backbone-depth', default=12, type=int)
    
    # Optimizer parameters
    parser.add_argument('--lr', default=1.0e-4, type=float)
    parser.add_argument('--max-epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=10, type=int)
    parser.add_argument('--eval-epochs', default=20, type=int)

    # Dataset parameters
    # Support datasets: NWPU_RESISC45 WHURS19
    parser.add_argument('--dataset', default='NWPU_RESISC45', type=str, help='dataset name')
    parser.add_argument('--batch-size', default=32, type=int)

    return parser.parse_args()


def main(args):
    myLogger = MY_Logger(args.output_dir)
    myLogger.logger.info(args)
    
    if args.seed is not None:
        myLogger.logger.info(f"set seed: {args.seed}")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        cudnn.benchmark = True
        
    dataset_name = args.dataset
    
    if dataset_name == 'NWPU_RESISC45':
        # build dataset
        dataset_cfg = NWPURESIS45Dataset.get_cfg()
        train_dataset = NWPURESIS45Dataset(metainfo_file=dataset_cfg['metainfo_train'], root_path=dataset_cfg['root_path'], is_train=True)
        test_dataset = NWPURESIS45Dataset(metainfo_file=dataset_cfg['metainfo_test'], root_path=dataset_cfg['root_path'], is_train=False)
        
        # build model
        tokenizer = VisTokenizer(img_size=224, in_channels=3, downsample_dim=args.downsample_dim, token_num=args.token_num, 
                                 out_dim=768, with_cls_token=False)
        backbone = AttnBlocks(depth=args.backbone_depth)
        taskHead = LinearClsHead(num_classes=45, in_channels=768, mode="mean")
        model = TokenModel(tokenizer, backbone, taskHead)
        model.cuda()
    
    elif dataset_name == 'WHURS19':
        # build dataset
        dataset_cfg = WHURS19Dataset.get_cfg()
        train_dataset = WHURS19Dataset(metainfo_file=dataset_cfg['metainfo_train'], root_path=dataset_cfg['root_path'], is_train=True)
        test_dataset = WHURS19Dataset(metainfo_file=dataset_cfg['metainfo_test'], root_path=dataset_cfg['root_path'], is_train=False)
        
        # build model
        tokenizer = VisTokenizer(img_size=224, in_channels=3, downsample_dim=args.downsample_dim, token_num=args.token_num, 
                                 out_dim=768, with_cls_token=False)
        backbone = AttnBlocks(depth=args.backbone_depth)
        taskHead = LinearClsHead(num_classes=19, in_channels=768, mode="mean")
        model = TokenModel(tokenizer, backbone, taskHead)
        model.cuda()
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.05)
    
    # summary params
    freeze_param, trainable_param = 0, 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_param += p.numel()
        else:
            freeze_param += p.numel()
            
    myLogger.logger.info("-" * 40)        
    myLogger.logger.info(f"### Total Params: {(freeze_param + trainable_param) / 1e6:.2f}M")
    myLogger.logger.info(f"### Freeze Params: {freeze_param / 1e6:.2f}M")
    myLogger.logger.info(f"### Trainable Params: {trainable_param / 1e6:.2f}M")
    myLogger.logger.info("-" * 40) 

    # run
    cur_iters = 0
    iters_per_epoch = len(train_loader)
    max_epoch = int(args.max_epochs)
    max_iters = max_epoch * iters_per_epoch
    warmup_iters = int(args.warmup_epochs) * iters_per_epoch
    print_loss = 0.
    run_time = Avg_values()
    
    # evaluation(model, test_loader, test_dataset, 0, args, myLogger=myLogger)
    for cur_epoch in range(max_epoch):
        model.train()
        for i, (image, label) in enumerate(train_loader):
            start_time = datetime.now()
            
            image = image.cuda()
            label = label.cuda()
            
            cur_iters += 1
            lr = adjust_learning_rate(optimizer, cur_iters, warmup_iters, max_iters, args.lr, 0.)
            
            with torch.cuda.amp.autocast(True):
                pred = model(image)
                loss = taskHead.loss(pred, label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 
        
            print_loss += loss.item()
            
            end_time = datetime.now()
            run_time.update(end_time-start_time, 1)
            eta_str = str(run_time.avg * (max_iters - run_time.count))
            
            if cur_iters % args.print_freq == 0:
                myLogger.logger.info(f"epoch:{cur_epoch+1} cur_iter:{i+1}/{iters_per_epoch} "
                                f"total_iters:{cur_iters} lr:{lr:.6e} loss:{print_loss / args.print_freq:.6f} eta:{eta_str}")
                print_loss = 0.
              
        if (cur_epoch+1) % args.eval_epochs == 0 or cur_epoch == max_epoch-1:
            evaluation(model, test_loader, test_dataset, cur_epoch+1, args, myLogger=myLogger)
            
    print("Done!")
    
@torch.no_grad()
def evaluation(model, test_loader, test_dataset, cur_epoch, args, myLogger):
    model.eval()
    
    save_path = os.path.join(args.output_dir, f"{cur_epoch}-epoch")
    
    run_time = Avg_values()
    test_iters = len(test_loader)
    for i, (image, label) in enumerate(test_loader):
        start_time = datetime.now()
        
        image = image.cuda()
        label = label.cuda()
        
        pred = model(image)
        test_dataset.eval(pred, label)
        if args.vis:
            model.save_attn(image)
        
        end_time = datetime.now()
        run_time.update(end_time-start_time, 1)
        eta_str = str(run_time.avg * (len(test_loader) - run_time.count))
        
        if i % args.print_freq == 0:
            myLogger.logger.info(f"[val] cur_iters:{i}/{test_iters} eta:{eta_str}")

    metrics = test_dataset.get_eval_res()    
    myLogger.logger.info("-" * 40)
    for key in metrics.keys():
        myLogger.logger.info(f"### {key}: {metrics[key]}")
    myLogger.logger.info("-" * 40)
    if args.vis:
        model.save_attn_to_file(save_path, cur_epoch)
    

if __name__ == "__main__":
    args = get_args_parser()
    if args.exp_name is None:
        print("It is recommended to specific your experiment name")
        args.exp_name = ""
        
    args.output_dir = os.path.join("./workdirs", args.exp_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)