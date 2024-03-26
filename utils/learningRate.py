import math
import numpy as np

def adjust_learning_rate(optimizer, cur_iters, warmup_iters, max_iters, base_lr, min_lr):
    if cur_iters < warmup_iters:
        lr = base_lr * cur_iters / warmup_iters
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * (cur_iters - warmup_iters) / (max_iters - warmup_iters)))
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
