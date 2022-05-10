import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensor, ToTensorV2
import pytorch_lightning as pl
from pytorch_lightning import Trainer

import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import BinaryDiceLoss, dice_coef_metric
from torchvision import transforms
from pytorch_toolbelt import losses as L
import matplotlib.pyplot as plt
from pytorch_toolbelt import losses as L
from pytorch_lightning.callbacks import ModelCheckpoint


def trainer_alveolar(args, model, output_dir, train_data_dir, test_data_dir):

    logging.basicConfig(filename=output_dir + "/log.txt",
                        level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    from datasets.dataset_alveolar import ImageTransforms, AlveolarDataset, AlveolarDataModule

    transform = ImageTransforms(img_size=448)
    dm = AlveolarDataModule(train_data_dir,
                            transform,
                            batch_size,
                            phase='train',
                            seed=args.seed)
    dm.prepare_data()
    dm.setup()
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    dm = AlveolarDataModule(test_data_dir,
                            transform,
                            batch_size,
                            phase='test',
                            seed=args.seed)
    dm.prepare_data()
    dm.setup()
    test_dataloader = dm.test_dataloader()

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()

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

            out_cut = np.copy(outputs.data.cpu().float().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

            train_dice = self.Compute_IoU(out_cut,
                                          target.data.cpu().float().numpy())
            loss = self.loss_fn(outputs, target)
            lr_ = self.lr * (1.0 -
                             self.train_iter_num / self.max_iterations)**0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_
            self.train_iter_num += 1
            print(' loss: ', loss.item(), ' IoU: ', train_dice)

            writer.add_scalar('info/lr', lr_, self.train_iter_num)
            writer.add_scalar('info/total_loss', loss.item(),
                              self.train_iter_num)
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

            print("Val_IoU: ", train_dice)
            self.log('Val_IoU', train_dice)
            return train_dice

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
            return train_dice
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

        def test_epoch_end(self, outputs):
            print(sum(outputs) / len(outputs))

    if args.pattern == 'train':
        model = AlveolarNet_lightningSystem(model, base_lr, args.max_epochs,
                                            len(train_dataloader), transform)
        checkpoint_callback = ModelCheckpoint(
            monitor='Val_IoU',
            dirpath='./output',
            filename='Alveolar-Dataset-{epoch:02d}-{Val_IoU:.2f}',
            mode='max')

        trainer = Trainer(
            logger=False,
            max_epochs=args.max_epochs,
            gpus=1,
            reload_dataloaders_every_n_epochs=False,
            num_sanity_val_steps=0,  # Skip Sanity Check
            callbacks=[checkpoint_callback],
            #precision=16,
            #accumulate_grad_batches=8,
            #gradient_clip_val=0.5,
        )

        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        model = AlveolarNet_lightningSystem.load_from_checkpoint(
            net=model,
            lr=base_lr,
            epoch=args.max_epochs,
            len=len(train_dataloader),
            transform=transform,
            checkpoint_path=
            './output/Alveolar-Dataset-epoch=13-Val_IoU=0.95.ckpt')
        trainer = Trainer(
            logger=False,
            gpus=1,
            #limit_test_batches=0.05,
            #precision=16,
            #accumulate_grad_batches=8,
            #gradient_clip_val=0.5,
        )
        trainer.test(model=model, dataloaders=test_dataloader)

    writer.close()

    return "Training Finished!"