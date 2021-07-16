import torch
import torch.nn.functional as F
import torch.nn as nn


class EdgeLoss2d(nn.Module):
    def __init__(self, edge_weight=1):
        super(EdgeLoss2d, self).__init__()
        self.edge_weight = edge_weight

    def bce2d(self, input, target):
        n, c, h, w = input.size()

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = (target_t == 1)
        neg_index = (target_t == 0)
        ignore_index = (target_t > 1)

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.1 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()
        loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight)
        return loss

    def forward(self, inputs, targets):
       
        loss = self.edge_weight * self.bce2d(inputs, targets)

        return loss
