import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def _iou(pred, target, gt_union,size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:][gt_union[i]]*pred[i,:,:,:][gt_union[i]])
        Ior1 = torch.sum(target[i,:,:,:][gt_union[i]]) + torch.sum(pred[i,:,:,:][gt_union[i]])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return (IoU/b)

class IOU(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target,gt_union):

        return _iou(pred, target, gt_union,self.size_average)
