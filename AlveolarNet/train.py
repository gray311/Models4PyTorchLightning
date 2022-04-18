import os, glob, random
from turtle import forward
from xml.sax.xmlreader import InputSource
from cv2 import accumulate, transform
from matplotlib.style import reload_library
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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

from AlveolarDataloader import ImageTransforms, AlveolarDataset, AlveolarDataModule
from model import Unet
from loss import dice_coef_metric, BinaryDiceLoss

from tensorboardX import SummaryWriter

print(pl.__version__)
from pytorch_toolbelt import losses as L


def seedeverything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  ##
    torch.backends.cudnn.benchmark = True


seed = 233
seedeverything(seed)

TrainData_dir = './train'
TestData_dir = './test'
output_dir = './'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter(output_dir + '/log')


class AlveolarNet_lightningSystem(pl.LightningModule):
    def __init__(self, net, lr, epoch, len, transform):
        super(AlveolarNet_lightningSystem, self).__init__()

        self.net = net.to(device)
        self.lr = lr
        self.epoch = epoch
        self.transform = transform
        self.step = 0
        self.train_iter_num = 0
        self.val_iter_num = 0
        self.test_iter_num = 0
        self.max_iterations = self.epoch * len

        self.Compute_IoU = dice_coef_metric()
        bce_loss = nn.BCEWithLogitsLoss()
        dice_loss = BinaryDiceLoss()
        self.loss_fn = L.JointLoss(first=dice_loss,
                                   second=bce_loss,
                                   first_weight=0.5,
                                   second_weight=0.5).cuda()

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(model.parameters(),
                                           lr=self.lr,
                                           weight_decay=1e-3)
        return self.optimizer

    def training_step(self, batch, batch_idx):
        img, mask = batch
        data, target = img.cuda(), mask.cuda()
        outputs = self.net(data)

        out_cut = np.copy(outputs.data.cpu().numpy())
        out_cut[np.nonzero(out_cut < 0.5)] = 0.0
        out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

        train_dice = self.Compute_IoU(out_cut, target.data.cpu().numpy())
        loss = self.loss_fn(outputs, target)
        lr_ = self.lr * (1.0 - self.train_iter_num / self.max_iterations)**0.9
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_
        self.train_iter_num += 1
        print(' loss: ', loss.item(), ' IoU: ', train_dice)

        writer.add_scalar('info/lr', lr_, self.train_iter_num)
        writer.add_scalar('info/total_loss', loss.item(), self.train_iter_num)
        writer.add_scalar('info/train_dice', train_dice.item(),
                          self.train_iter_num)

        if self.train_iter_num % 20 == 0:

            image = data[0, :, :, :].permute(1, 2, 0).data.cpu().numpy()
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image = image * std + mean
            image = image * 255.0
            image = (image - image.min()) / (image.max() - image.min())
            writer.add_image('train/Image', image.transpose(2, 0, 1),
                             self.train_iter_num)

            pred = out_cut[0, :, :, :] * 50
            writer.add_image('train/Prediction', pred, self.train_iter_num)
            gt = target[0, :, :, :] * 50
            writer.add_image('train/GroundTruth', gt, self.train_iter_num)

        return {'loss': loss, 'IoU': train_dice}

    def validation_step(self, batch, batch_idx):
        img, mask = batch
        data, target = img.cuda(), mask.cuda()
        outputs = self.net(data)

        out_cut = np.copy(outputs.data.cpu().numpy())
        out_cut[np.nonzero(out_cut < 0.5)] = 0.0
        out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

        train_dice = self.Compute_IoU(out_cut, target.data.cpu().numpy())
        print("val_IoU: ", train_dice)

        self.val_iter_num += 1

        if self.val_iter_num % 10 == 0:

            image = data[0, :, :, :].permute(1, 2, 0).data.cpu().numpy()
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image = image * std + mean
            image = image * 255.0
            image = (image - image.min()) / (image.max() - image.min())
            writer.add_image('val/Image', image.transpose(2, 0, 1),
                             self.val_iter_num)

            pred = out_cut[0, :, :, :] * 50
            writer.add_image('val/Prediction', pred, self.val_iter_num)
            gt = target[0, :, :, :] * 50
            writer.add_image('val/GroundTruth', gt, self.val_iter_num)

    def test_step(self, batch, batch_idx):
        img, mask = batch
        data, target = img.cuda(), mask.cuda()
        outputs = self.net(data)

        out_cut = np.copy(outputs.data.cpu().numpy())
        out_cut[np.nonzero(out_cut < 0.5)] = 0.0
        out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

        train_dice = self.Compute_IoU(out_cut, target.data.cpu().numpy())
        print("test_IoU: ", train_dice)

        self.test_iter_num += 1

        if self.test_iter_num % 10 == 0:

            image = data[0, :, :, :].permute(1, 2, 0).data.cpu().numpy()
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image = image * std + mean
            image = image * 255.0
            image = (image - image.min()) / (image.max() - image.min())
            writer.add_image('test/Image', image.transpose(2, 0, 1),
                             self.test_iter_num)

            pred = out_cut[0, :, :, :] * 50
            writer.add_image('test/Prediction', pred, self.test_iter_num)
            gt = target[0, :, :, :] * 50
            writer.add_image('test/GroundTruth', gt, self.test_iter_num)

    def training_epoch_end(self, outputs):

        self.step += 1
        outputs = np.array(outputs)
        print(outputs.shape)


transform = ImageTransforms(img_size=448)
batch_size = 4
lr = 5e-4
epoch = 20

dm = AlveolarDataModule(TrainData_dir,
                        transform,
                        batch_size,
                        phase='train',
                        seed=seed)
dm.prepare_data()
dm.setup()
train_dataloader = dm.train_dataloader()
val_dataloader = dm.val_dataloader()

dm = AlveolarDataModule(TestData_dir,
                        transform,
                        batch_size,
                        phase='test',
                        seed=seed)
dm.prepare_data()
dm.setup()
test_dataloader = dm.test_dataloader()

net = Unet()
model = AlveolarNet_lightningSystem(net, lr, epoch, len(train_dataloader),
                                    transform)

trainer = Trainer(
    logger=False,
    max_epochs=epoch,
    gpus=1,
    enable_checkpointing=False,
    reload_dataloaders_every_epoch=False,
    reload_dataloaders_every_n_epochs=False,
    num_sanity_val_steps=0,  # Skip Sanity Check
)

trainer.fit(model, train_dataloader, val_dataloader)
trainer.test(model, ckpt_path="best", test_dataloaders=test_dataloader)
writer.close()
