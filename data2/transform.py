#!/usr/bin/python3
# coding=utf-8

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms


class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, mask, edge,region):
        for op in self.ops:
            image, mask, edge,region = op(image, mask, edge,region)
        return image, mask, edge,region


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, edge,region):
        image = (image - self.mean) / self.std
        mask /= 255
        edge /= 255
        region /= 255
        return image, mask, edge,region


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, edge,region):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        edge = cv2.resize(edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        region = cv2.resize(region, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, edge,region


class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, edge,region):
        H, W, _ = image.shape
        xmin = np.random.randint(W - self.W + 1)
        ymin = np.random.randint(H - self.H + 1)
        image = image[ymin:ymin + self.H, xmin:xmin + self.W, :]
        mask = mask[ymin:ymin + self.H, xmin:xmin + self.W, :]
        edge = edge[ymin:ymin + self.H, xmin:xmin + self.W, :]
        region= region[ymin:ymin + self.H, xmin:xmin + self.W, :]
        return image, mask, edge,region


class RandomHorizontalFlip(object):
    def __call__(self, image, mask, edge,region):
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
            edge = edge[:, ::-1, :].copy()
            region = region[:, ::-1, :].copy()
        return image, mask, edge,region


class ToTensor(object):
    def __call__(self, image, mask, edge,region):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        mask = mask.permute(2, 0, 1)
        edge = torch.from_numpy(edge)
        edge = edge.permute(2, 0, 1)
        region = torch.from_numpy(region)
        region = region.permute(2, 0, 1)
        return image, mask.mean(dim=0, keepdim=True), edge.mean(dim=0, keepdim=True),region.mean(dim=0, keepdim=True)
# class colorj(object):
#     def __call__(self,image, mask, edge,region):
#         color_aug = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
#         return color_aug

