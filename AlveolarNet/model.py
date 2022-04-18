import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, glob, random
import cv2
import torch
from torchvision.utils import make_grid
from torchvision import transforms
from torch import batch_norm, conv2d, nn, optim, relu
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import albumentations as A
from albumentations.pytorch import ToTensor, ToTensorV2


def seedeverything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  ##
    torch.backends.cudnn.benchmark = True


seedeverything(seed=233)


class DownSample(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(DownSample, self).__init__()
        if channel_in == 3:
            self.block = nn.Sequential(
                nn.Conv2d(channel_in,
                          channel_out,
                          kernel_size,
                          stride,
                          padding,
                          bias=False),
                nn.BatchNorm2d(channel_out), nn.ReLU(inplace=True),
                nn.Conv2d(channel_out,
                          channel_out,
                          kernel_size,
                          stride,
                          padding,
                          bias=False), nn.BatchNorm2d(channel_out),
                nn.ReLU(inplace=True))
        else:
            self.block = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(channel_in,
                          channel_out,
                          kernel_size,
                          stride,
                          padding,
                          bias=False), nn.BatchNorm2d(channel_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel_out,
                          channel_out,
                          kernel_size,
                          stride,
                          padding,
                          bias=False), nn.BatchNorm2d(channel_out),
                nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class UpSample(nn.Module):
    def __init__(self,
                 channel_in,
                 channel_out,
                 kernel_size=4,
                 stride=2,
                 padding=1):
        super(UpSample, self).__init__()
        self.block = nn.ModuleList([
            nn.ConvTranspose2d(channel_in,
                               channel_out,
                               kernel_size,
                               stride,
                               padding,
                               bias=False),
            nn.BatchNorm2d(channel_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_in,
                      channel_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
            nn.Conv2d(channel_out,
                      channel_out,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(channel_out),
            nn.ReLU()
        ])

    def forward(self, x, shortcut=None):
        for layer in self.block:

            x = layer(x)
            if isinstance(layer, nn.ConvTranspose2d) is True:
                if shortcut is not None:

                    x = torch.cat([x, shortcut], dim=1)

        return x


class Unet(nn.Module):
    def __init__(self, filter=64):
        super(Unet, self).__init__()
        self.downsamples = nn.ModuleList([
            DownSample(3, filter),
            DownSample(filter, filter * 2),
            DownSample(filter * 2, filter * 4),
            DownSample(filter * 4, filter * 8),
            DownSample(filter * 8, filter * 16),
        ])

        self.upsamples = nn.ModuleList([
            UpSample(filter * 16, filter * 8),
            UpSample(filter * 8, filter * 4),
            UpSample(filter * 4, filter * 2),
            UpSample(filter * 2, filter),
        ])

        self.last = nn.Sequential(
            nn.Conv2d(in_channels=filter,
                      out_channels=1,
                      kernel_size=1,
                      stride=1,
                      padding=0), nn.Tanh())

    def forward(self, x):
        skips = []
        w, h = x.shape[2:]
        for layer in self.downsamples:
            x = layer(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        for layer, shortcut in zip(self.upsamples, skips):
            x = layer(x, shortcut)
        out = self.last(x)
        return out


'''
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
input = torch.ones(2, 3, 576, 576).to(device)
net = Unet().to(device)
print(net(input).shape)
'''
