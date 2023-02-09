from torch.utils.data import dataset

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F

import h5py


class LoadData(dataset.Dataset):
    def __init__(self, path, train=True):
        super(LoadData, self).__init__()
        self.train_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.test_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        f = h5py.File(path)
        self.data = f['rgb']  # type dataset
        self.label = f['seg']  # inside type nparray
        self.train = train

    def __getitem__(self, index):
        print(index)
        img = self.data[index]
        label = self.label[index]
        if self.train:
            img = self.train_transform(img)
            label = self.train_transform(label)
        else:
            img = self.test_transform(img)
            label = self.test_transform(label)
        return img, label

    def __len__(self):
        return self.data.shape[0]
