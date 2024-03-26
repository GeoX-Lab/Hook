import argparse
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from terminaltables import AsciiTable
from torch.backends import cudnn
from torch.utils.data import DataLoader

from models import (AttnBlocks, DenseTransform, S2DAdapter, SegFormerHead,
                    TokenModel_Forseg, VisTokenizer)
from token_datasets import GIDDataset
from utils import Avg_values, MY_Logger, adjust_learning_rate, seg_metrics


def get_args_parser():
    parser = argparse.ArgumentParser('Visual Tokenizer', add_help=False)
    
    # Common parameters
    parser.add_argument('--exp-name', default="debug", type=str, help='current experiment name')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--print-freq', default=20, type=int)
    
    # Model parameters
    parser.add_argument('--token-num', default=6, type=int)
    parser.add_argument('--vis', action="store_true")
    parser.add_argument('--downsample-dim', default=512, type=int)
    parser.add_argument('--backbone-depth', default=12, type=int)
    parser.add_argument('--out-layers', default=[2, 5, 9, 11], type=int, nargs="+")
    
    # Optimizer parameters
    parser.add_argument('--lr', default=5.0e-6, type=float)
    parser.add_argument('--max-epochs', default=100, type=int)
    parser.add_argument('--warmup-epochs', default=10, type=int)
    parser.add_argument('--eval-epochs', default=20, type=int)

    # Dataset parameters
    # Support datasets: GID
    parser.add_argument('--dataset', default='GID', type=str, help='dataset name')
    parser.add_argument('--batch-size', default=8, type=int)

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
    
    if dataset_name == 'GID':
        # build dataset
        dataset_cfg = GIDDataset.get_cfg()
        train_dataset = GIDDataset(img_dir=dataset_cfg['train_imgdir'], ann_dir=dataset_cfg['train_anndir'], is_train=True)
        test_dataset = GIDDataset(img_dir=dataset_cfg['test_imgdir'], ann_dir=dataset_cfg['test_anndir'], is_train=False)
        
        # build model
        tokenizer = VisTokenizer(img_size=512, in_channels=3, downsample_dim=args.downsample_dim, token_num=args.token_num, 
                                 out_dim=768, with_cls_token=False)
        backbone = AttnBlocks(depth=args.backbone_depth, ret_layers=args.out_layers)
        s2dAdapter = S2DAdapter(32)
        denseTransform = DenseTransform(embed_dim=768, out_dim_list=[128, 256, 512, 512])
        taskHead = SegFormerHead(num_classes=train_dataset.num_classes, in_channels=[128, 256, 512, 512], embedding_dim=256, ignore_index=train_dataset.ignore_idx)
        model = TokenModel_Forseg(tokenizer, backbone, s2dAdapter, denseTransform, taskHead)
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
            label = label.long().cuda()
            
            cur_iters += 1
            lr = adjust_learning_rate(optimizer, cur_iters, 0, max_iters, args.lr, 0.)
            
            with torch.cuda.amp.autocast(True):
                loss = model(image, label, is_train=True)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad() 
        
            print_loss += loss.item()
            
            end_time = datetime.now()
            run_time.update(end_time-start_time, 1)
            eta_str = str(run_time.avg * (max_iters - run_time.count))
            
            if cur_iters % args.print_freq == 0 and cur_iters != 0:
                myLogger.logger.info(f"epoch:{cur_epoch+1} cur_iter:{i+1}/{iters_per_epoch} "
                                f"total_iters:{cur_iters} lr:{lr:.6e} loss:{print_loss / args.print_freq:.6f} eta:{eta_str}")
                print_loss = 0.
              
        if (cur_epoch+1) % args.eval_epochs == 0 or cur_epoch == max_epoch-1:
            evaluation(model, test_loader, test_dataset, cur_epoch+1, args, myLogger=myLogger)
            
    print("Done!")
    
@torch.no_grad()
def evaluation(model, test_loader, test_dataset, cur_epoch, args, myLogger):
    model.eval()
    
    args.vis = True
    attn_save_path = os.path.join(args.output_dir, f"{cur_epoch}-epoch-attn")
    seg_save_path = os.path.join(args.output_dir, f"{cur_epoch}-epoch-seg")
    
    imgs, preds, labels = [], [], []
    
    run_time = Avg_values()
    test_iters = len(test_loader)
    for i, (image, label) in enumerate(test_loader):
        start_time = datetime.now()
        
        image = image.cuda()
        label = label.long().cuda()
        
        pred = model(image, label, is_train=False)
        
        imgs.append(image.detach().cpu())
        preds.append(pred.detach().cpu())
        labels.append(label.detach().cpu())
        
        if args.vis:
            model.save_attn(image)
        
        end_time = datetime.now()
        run_time.update(end_time-start_time, 1)
        eta_str = str(run_time.avg * (len(test_loader) - run_time.count))
        
        if i % args.print_freq == 0 and i != 0:
            myLogger.logger.info(f"[val] cur_iters:{i}/{test_iters} eta:{eta_str}")
    
    imgs = torch.concatenate(imgs, dim=0)
    preds = torch.concatenate(preds, dim=0)
    labels = torch.concatenate(labels, dim=0)

    metrics = seg_metrics(preds, labels, num_classes=test_dataset.num_classes, 
                          ignore_index=test_dataset.ignore_idx)
    class_table_data = [['Class'] + ['IoU'] + ['Acc']]
    if test_dataset.CLASSES is None:
        class_names = tuple(range(test_dataset.num_classes))
    else:
        class_names = test_dataset.CLASSES
    ret_metrics_round = [
        np.round(ret_metric * 100, 2) for ret_metric in metrics
    ]
    for i in range(test_dataset.num_classes):
        class_table_data.append([class_names[i]] +
                                [m[i] for m in ret_metrics_round[2:]] +
                                [ret_metrics_round[1][i]])
    summary_table_data = [['Scope'] +
                            ['m' + head
                            for head in class_table_data[0][1:]] + ['aAcc']]
    ret_metrics_mean = [
        np.round(np.nanmean(ret_metric) * 100, 2)
        for ret_metric in metrics
    ]
    summary_table_data.append(['global'] + ret_metrics_mean[2:] +
                                [ret_metrics_mean[1]] +
                                [ret_metrics_mean[0]])
    myLogger.logger.info("-" * 40)
    myLogger.logger.info('per class results:')
    table = AsciiTable(class_table_data)
    myLogger.logger.info('\n' + table.table)
    myLogger.logger.info('Summary:')
    table = AsciiTable(summary_table_data)
    myLogger.logger.info('\n' + table.table)
    myLogger.logger.info("-" * 40)
    if args.vis:
        model.save_attn_to_file(attn_save_path, cur_epoch)
        model.save_segmap(imgs, labels, preds, test_dataset.PALETTE, seg_save_path)
    

if __name__ == "__main__":
    args = get_args_parser()
    if args.exp_name is None:
        print("It is recommended to specific your experiment name")
        args.exp_name = ""
        
    args.output_dir = os.path.join("./workdirs", args.exp_name, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)