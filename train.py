#!/usr/bin/python3
#coding=utf-8

import sys
import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data2 import dataset
from net import GFINet
import logging as logger
from lib2.data_prefetcher import DataPrefetcher
import numpy as np
import matplotlib.pyplot as plt
import pytorch_iou2
from edge_loss import EdgeLoss2d

TAG = "ours"
SAVE_PATH = "ours"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")


""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum

BASE_LR = 1e-3
MAX_LR = 0.1


# ------- 1. define loss function --------

bce_loss = nn.BCELoss(size_average=True)
iou_loss2 = pytorch_iou2.IOU(size_average=True)
edge_loss=EdgeLoss2d(edge_weight=1)

def bce_ssim_loss2(pred,target,gt_union):
    iou_out = iou_loss2(F.sigmoid(pred),target,gt_union)
    return iou_out

def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode='train', batch=24, lr=0.05, momen=0.9, decay=5e-4, epoch=40)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=4)
    prefetcher = DataPrefetcher(loader)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer   = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    global_step = 0
    db_size = len(loader)

    #training
    for epoch in range(cfg.epoch):
        prefetcher = DataPrefetcher(loader)
        batch_idx = -1
        image, mask,edge,edge_area = prefetcher.next()
        while image is not None:
            niter = epoch * db_size + batch_idx
            lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch*db_size, niter, ratio=1.)
            optimizer.param_groups[0]['lr'] = 0.1 * lr #for backbone
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum = momentum
           
            batch_idx += 1
            global_step += 1
            
            gt_edge_area=(edge_area!=0)#thicker edge           
            
            gt_n_foreground=(mask!=0)
            gt_union=(gt_edge_area | gt_n_foreground )

            gt_background=(gt_union==False)  

            back_num=np.sum((gt_background.cpu().numpy()!=0))
            fore_edge_area_union_num=np.sum((gt_union.cpu().numpy()!=0))
        
            n=fore_edge_area_union_num+back_num

            if fore_edge_area_union_num<back_num:
                num_f=back_num/n
                num_b=fore_edge_area_union_num/n
                              
            else:
                num_b=back_num/n
                num_f=fore_edge_area_union_num/n

            out2, out3, out4, out5,edge_predict= net(image)
            loss2                  = F.binary_cross_entropy_with_logits(out2[gt_union], mask[gt_union])+bce_ssim_loss2(out2,mask,gt_union)
            loss3                  = F.binary_cross_entropy_with_logits(out3[gt_union], mask[gt_union])+bce_ssim_loss2(out3,mask,gt_union)
            loss4                  = F.binary_cross_entropy_with_logits(out4[gt_union], mask[gt_union])+bce_ssim_loss2(out4,mask,gt_union)
            loss5                  = F.binary_cross_entropy_with_logits(out5[gt_union], mask[gt_union])+bce_ssim_loss2(out5,mask,gt_union)
            loss_fore              = loss2*1 + loss3*0.8 + loss4*0.6 + loss5*0.4
  
            loss22                  = F.binary_cross_entropy_with_logits(out2[gt_background], mask[gt_background])+bce_ssim_loss2(out2,mask,gt_background)
            loss33                  = F.binary_cross_entropy_with_logits(out3[gt_background], mask[gt_background])+bce_ssim_loss2(out3,mask,gt_background)
            loss44                  = F.binary_cross_entropy_with_logits(out4[gt_background], mask[gt_background])+bce_ssim_loss2(out4,mask,gt_background)
            loss55                  = F.binary_cross_entropy_with_logits(out5[gt_background], mask[gt_background])+bce_ssim_loss2(out5,mask,gt_background)
            loss_back              = loss22*1 + loss33*0.8 + loss44*0.6 + loss55*0.4
           
            loss_edge             =edge_loss(edge_predict, edge)            

            loss=num_f*loss_fore+num_b*loss_back+10*loss_edge
            optimizer.zero_grad()
            loss.backward()   
            optimizer.step()
           
            if batch_idx % 10 == 0:               
                msg = '%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss_fore=%.6f | loss_back=%.6f | loss_edge=%.6f'%(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item(), loss_fore.item(), loss_back.item(),10*loss_edge.item())
                print(msg)
                logger.info(msg)
            image, mask,edge,edge_area = prefetcher.next()

        if epoch>cfg.epoch/5*4:
            torch.save(net.state_dict(), cfg.savepath+'/model24_'+str(epoch+1))



if __name__=='__main__':
    train(dataset, GFINet)
