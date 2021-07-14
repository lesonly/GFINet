#!/usr/bin/python3
# coding=utf-8

import cv2
import torch
import numpy as np


class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, image, mask, edge):
        for op in self.ops:
            image, mask, edge = op(image, mask, edge)
        return image, mask, edge


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, edge):
        image = (image - self.mean) / self.std
        mask /= 255
        edge /= 255
        return image, mask, edge


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, edge):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        edge = cv2.resize(edge, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, edge


class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, edge):
        H, W, _ = image.shape
        xmin = np.random.randint(W - self.W + 1)
        ymin = np.random.randint(H - self.H + 1)
        image = image[ymin:ymin + self.H, xmin:xmin + self.W, :]
        mask = mask[ymin:ymin + self.H, xmin:xmin + self.W, :]
        edge = edge[ymin:ymin + self.H, xmin:xmin + self.W, :]
        return image, mask, edge


class RandomHorizontalFlip(object):
    def __call__(self, image, mask, edge):
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
            edge = edge[:, ::-1, :].copy()
        return image, mask, edge


class ToTensor(object):
    def __call__(self, image, mask, edge):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        mask = mask.permute(2, 0, 1)
        edge = torch.from_numpy(edge)
        edge = edge.permute(2, 0, 1)
        return image, mask.mean(dim=0, keepdim=True), edge.mean(dim=0, keepdim=True)
