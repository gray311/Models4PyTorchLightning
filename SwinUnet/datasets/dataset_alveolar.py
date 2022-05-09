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
from sklearn.model_selection import train_test_split


def seedeverything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  ##
    torch.backends.cudnn.benchmark = True


seedeverything(seed=233)


class ImageTransforms:

    def __init__(self, img_size):
        super(ImageTransforms, self).__init__()
        self.transforms = {
            "train":
            A.Compose([
                A.Resize(width=img_size, height=img_size, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.01,
                                   scale_limit=0.04,
                                   rotate_limit=0,
                                   p=0.25),
                A.Normalize(p=1.0),
                ToTensor(),
            ]),
            "test":
            A.Compose([
                A.Resize(width=img_size, height=img_size, p=1.0),
                A.Normalize(p=1.0),
                ToTensor(),
            ])
        }

    def __call__(self, img, mask, phase='train'):
        return self.transforms[phase](image=img, mask=mask)


class AlveolarDataset(Dataset):

    def __init__(self, img_dir, mask_dir, transforms=None, phase='train'):
        super(AlveolarDataset, self).__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.phase = phase

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        self.img_path = self.img_dir[idx]
        self.mask_path = self.mask_dir[idx]
        self.img = cv2.imread(self.img_path)
        self.mask = cv2.imread(self.mask_path, 0)

        augmented = self.transforms(self.img, self.mask, self.phase)

        self.img = augmented['image']
        self.mask = augmented['mask']

        return augmented['image'], augmented['mask']


class AlveolarDataModule(pl.LightningDataModule):

    def __init__(self,
                 data_dir,
                 transforms=None,
                 batch_size=8,
                 phase='train',
                 seed=0):
        super(AlveolarDataModule, self).__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.seed = seed
        self.phase = phase
        self.batch_size = batch_size

    def prepare_data(self):

        temp = [
            os.path.join(self.data_dir, case, 'img', '*.png')
            for case in os.listdir(self.data_dir)
        ]
        self.img_dir = []
        for path in temp:
            self.img_dir += glob.glob(path)

        temp = [
            os.path.join(self.data_dir, case, 'mask', '*.png')
            for case in os.listdir(self.data_dir)
        ]
        self.mask_dir = []
        for path in temp:
            self.mask_dir += glob.glob(path)

    def setup(self, stage=None):
        if self.phase == 'train':
            self.train_img_dir, self.val_img_dir, self.train_mask_dir, self.val_mask_dir = train_test_split(
                self.img_dir,
                self.mask_dir,
                test_size=0.2,
                random_state=self.seed)
            self.train_dataset = AlveolarDataset(self.train_img_dir,
                                                 self.train_mask_dir,
                                                 self.transforms, self.phase)
            self.val_dataset = AlveolarDataset(self.val_img_dir,
                                               self.val_mask_dir,
                                               self.transforms, self.phase)
        if self.phase == 'test':
            self.test_dataset = AlveolarDataset(self.img_dir, self.mask_dir,
                                                self.transforms, self.phase)

    def train_dataloader(self):

        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):

        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True)


'''
data_dir = './train'
batch_size = 8
TransForms = ImageTransforms(img_size=256)

dm = AlveolarDataModule(data_dir, TransForms, batch_size, phase='train')
dm.prepare_data()
dm.setup()
train_dataloader = dm.train_dataloader()
val_dataloader = dm.val_dataloader()

img, mask = next(iter(train_dataloader))

print(img.shape, "    ", mask.shape)

img, mask = next(iter(val_dataloader))

print(img.shape, "    ", mask.shape)

temp = make_grid(img, nrow=4, padding=2).permute(1, 2, 0).detach().numpy()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
temp = temp * std + mean
temp = temp * 255.0
temp = temp.astype(int)

fig = plt.figure(figsize=(18, 8), facecolor='w')
plt.imshow(temp)
plt.axis('off')
plt.title('Photo')
plt.show()

temp = make_grid(mask, nrow=4, padding=2).permute(1, 2, 0).detach().numpy()
temp = temp * 255.0
temp = temp.astype(int)

fig = plt.figure(figsize=(18, 8), facecolor='w')
plt.imshow(temp)
plt.axis('off')
plt.title('Monet Pictures')
plt.show()
'''