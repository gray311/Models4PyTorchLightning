import os, glob, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil

import torch
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

# %%
pl.__version__


# %%
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# %% [markdown]
# ---
# ## Dataset


# %%
class ImageTransform:
    def __init__(self, img_size=256):
        self.transform = {
            'train':
            transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ]),
            'test':
            transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        }

    def __call__(self, img, phase='train'):
        img = self.transform[phase](img)
        return img


# Monet Dataset ---------------------------------------------------------------------------
class MonetDataset(Dataset):
    def __init__(self,
                 base_img_paths,
                 style_img_paths,
                 transform,
                 phase='train'):
        self.base_img_paths = base_img_paths
        self.style_img_paths = style_img_paths
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return min([len(self.base_img_paths), len(self.style_img_paths)])

    def __getitem__(self, idx):
        base_img_path = self.base_img_paths[idx]
        style_img_path = self.style_img_paths[idx]
        base_img = Image.open(base_img_path)
        style_img = Image.open(style_img_path)

        base_img = self.transform(base_img, self.phase)
        style_img = self.transform(style_img, self.phase)

        return base_img, style_img


# %%
# Data Module
class MonetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, transform, batch_size, phase='train', seed=0):
        super(MonetDataModule, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.batch_size = batch_size
        self.phase = phase
        self.seed = seed

    def prepare_data(self):
        self.base_img_paths = glob.glob(
            os.path.join(self.data_dir, 'photo_jpg', '*.jpg'))
        self.style_img_paths = glob.glob(
            os.path.join(self.data_dir, 'monet_jpg', '*.jpg'))

    def train_dataloader(self):
        random.seed()
        random.shuffle(self.base_img_paths)
        random.shuffle(self.style_img_paths)
        random.seed(self.seed)
        self.train_dataset = MonetDataset(self.base_img_paths,
                                          self.style_img_paths, self.transform,
                                          self.phase)

        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory=True)


# %%
# Sanity Check
data_dir = './gan-getting-started'
transform = ImageTransform(img_size=256)
batch_size = 16

dm = MonetDataModule(data_dir, transform, batch_size, phase='test')
dm.prepare_data()

dataloader = dm.train_dataloader()
base, style = next(iter(dataloader))

print('Input Shape {}, {}'.format(base.size(), style.size()))

# %%
temp = make_grid(base, nrow=4, padding=2).permute(1, 2, 0).detach().numpy()
temp = temp * 0.5 + 0.5
temp = temp * 255.0
temp = temp.astype(int)

fig = plt.figure(figsize=(18, 8), facecolor='w')
plt.imshow(temp)
plt.axis('off')
plt.title('Photo')
plt.show()

# %%
temp = make_grid(style, nrow=4, padding=2).permute(1, 2, 0).detach().numpy()
temp = temp * 0.5 + 0.5
temp = temp * 255.0
temp = temp.astype(int)

fig = plt.figure(figsize=(18, 8), facecolor='w')
plt.imshow(temp)
plt.axis('off')
plt.title('Monet Pictures')
plt.show()

# %% [markdown]
# ---
# ## Model


# %%
class Upsample(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 dropout=True):
        super(Upsample, self).__init__()
        self.dropout = dropout
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding,
                               bias=nn.InstanceNorm2d),
            nn.InstanceNorm2d(out_channels), nn.ReLU(inplace=True))
        self.dropout_layer = nn.Dropout2d(0.5)

    def forward(self, x, shortcut=None):
        x = self.block(x)
        if self.dropout:
            x = self.dropout_layer(x)

        if shortcut is not None:
            x = torch.cat([x, shortcut], dim=1)  #skip-connection

        return x


class Downsample(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=4,
                 stride=2,
                 padding=1,
                 apply_instancenorm=True):
        super(Downsample, self).__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              bias=nn.InstanceNorm2d)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.apply_norm = apply_instancenorm

    def forward(self, x):
        x = self.conv(x)
        if self.apply_norm:
            x = self.norm(x)
        x = self.relu(x)

        return x


class CycleGAN_Unet_Generator(nn.Module):
    def __init__(self, filter=64):
        super(CycleGAN_Unet_Generator, self).__init__()
        self.downsamples = nn.ModuleList([
            Downsample(3, filter, kernel_size=4,
                       apply_instancenorm=False),  # (b, filter, 128, 128)
            Downsample(filter, filter * 2),  # (b, filter * 2, 64, 64)
            Downsample(filter * 2, filter * 4),  # (b, filter * 4, 32, 32)
            Downsample(filter * 4, filter * 8),  # (b, filter * 8, 16, 16)
            Downsample(filter * 8, filter * 8),  # (b, filter * 8, 8, 8)
            Downsample(filter * 8, filter * 8),  # (b, filter * 8, 4, 4)
            Downsample(filter * 8, filter * 8),  # (b, filter * 8, 2, 2)
        ])

        self.upsamples = nn.ModuleList([
            Upsample(filter * 8, filter * 8),
            Upsample(filter * 16, filter * 8),  #有skip-connection
            Upsample(filter * 16, filter * 8),
            Upsample(filter * 16, filter * 4, dropout=False),
            Upsample(filter * 8, filter * 2, dropout=False),
            Upsample(filter * 4, filter, dropout=False)
        ])

        self.last = nn.Sequential(
            nn.ConvTranspose2d(filter * 2,
                               3,
                               kernel_size=4,
                               stride=2,
                               padding=1), nn.Tanh())

    def forward(self, x):
        skips = []
        for l in self.downsamples:
            x = l(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for l, s in zip(self.upsamples, skips):
            x = l(x, s)

        out = self.last(x)

        return out


class CycleGAN_Discriminator(nn.Module):
    def __init__(self, filter=64):
        super(CycleGAN_Discriminator, self).__init__()

        self.block = nn.Sequential(
            Downsample(3,
                       filter,
                       kernel_size=4,
                       stride=2,
                       apply_instancenorm=False),
            Downsample(filter, filter * 2, kernel_size=4, stride=2),
            Downsample(filter * 2, filter * 4, kernel_size=4, stride=2),
            Downsample(filter * 4, filter * 8, kernel_size=4, stride=1),
        )

        self.last = nn.Conv2d(filter * 8,
                              1,
                              kernel_size=4,
                              stride=1,
                              padding=1)

    def forward(self, x):
        x = self.block(x)
        x = self.last(x)

        return x


# %%
# Sanity Check
net = CycleGAN_Unet_Generator()

out = net(base)
print(out.size())

# %%
# Sanity Check
net = CycleGAN_Discriminator()

out = net(base)
print(out.size())


class CycleGAN_LightningSystem(pl.LightningModule):
    def __init__(self,
                 G_basestyle,
                 G_stylebase,
                 D_base,
                 D_style,
                 lr,
                 transform,
                 reconstr_w=10,
                 id_w=2):
        super(CycleGAN_LightningSystem, self).__init__()
        self.G_basestyle = G_basestyle
        self.G_stylebase = G_stylebase
        self.D_base = D_base
        self.D_style = D_style
        self.lr = lr
        self.transform = transform
        self.reconstr_w = reconstr_w
        self.id_w = id_w
        self.cnt_train_step = 0
        self.step = 0

        self.mae = nn.L1Loss()
        self.generator_loss = nn.MSELoss()
        self.discriminator_loss = nn.MSELoss()
        self.losses = []
        self.G_mean_losses = []
        self.D_mean_losses = []
        self.validity = []
        self.reconstr = []
        self.identity = []

    def configure_optimizers(self):
        self.g_basestyle_optimizer = optim.Adam(self.G_basestyle.parameters(),
                                                lr=self.lr['G'],
                                                betas=(0.5, 0.999))
        self.g_stylebase_optimizer = optim.Adam(self.G_stylebase.parameters(),
                                                lr=self.lr['G'],
                                                betas=(0.5, 0.999))
        self.d_base_optimizer = optim.Adam(self.D_base.parameters(),
                                           lr=self.lr['D'],
                                           betas=(0.5, 0.999))
        self.d_style_optimizer = optim.Adam(self.D_style.parameters(),
                                            lr=self.lr['D'],
                                            betas=(0.5, 0.999))

        return [
            self.g_basestyle_optimizer, self.g_stylebase_optimizer,
            self.d_base_optimizer, self.d_style_optimizer
        ], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        base_img, style_img = batch
        b = base_img.size()[0]

        valid = torch.ones(b, 1, 30, 30).cuda()
        fake = torch.zeros(b, 1, 30, 30).cuda()

        # Train Generator
        if optimizer_idx == 0 or optimizer_idx == 1:
            # Validity
            # MSELoss
            val_base = self.generator_loss(
                self.D_base(self.G_stylebase(style_img)), valid)
            val_style = self.generator_loss(
                self.D_style(self.G_basestyle(base_img)), valid)
            val_loss = (val_base + val_style) / 2
            # Reconstruction
            reconstr_base = self.mae(
                self.G_stylebase(self.G_basestyle(base_img)), base_img)
            reconstr_style = self.mae(
                self.G_basestyle(self.G_stylebase(style_img)), style_img)
            reconstr_loss = (reconstr_base + reconstr_style) / 2
            # Identity
            id_base = self.mae(self.G_stylebase(base_img), base_img)
            id_style = self.mae(self.G_basestyle(style_img), style_img)
            id_loss = (id_base + id_style) / 2
            # Loss Weight
            G_loss = val_loss + self.reconstr_w * reconstr_loss + self.id_w * id_loss

            return {
                'loss': G_loss,
                'validity': val_loss.detach(),
                'reconstr': reconstr_loss.detach(),
                'identity': id_loss.detach()
            }

        # Train Discriminator
        elif optimizer_idx == 2 or optimizer_idx == 3:
            # MSELoss
            D_base_gen_loss = self.discriminator_loss(
                self.D_base(self.G_stylebase(style_img)), fake)
            D_style_gen_loss = self.discriminator_loss(
                self.D_style(self.G_basestyle(base_img)), fake)
            D_base_valid_loss = self.discriminator_loss(
                self.D_base(base_img), valid)
            D_style_valid_loss = self.discriminator_loss(
                self.D_style(style_img), valid)
            D_gen_loss = (D_base_gen_loss + D_style_gen_loss) / 2
            # Loss Weight
            D_loss = (D_gen_loss + D_base_valid_loss + D_style_valid_loss) / 3
            # Count up
            self.cnt_train_step += 1
            return {'loss': D_loss}

    def training_epoch_end(self, outputs):
        self.step += 1

        avg_loss = sum([
            torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 4
            for i in range(4)
        ])
        G_mean_loss = sum([
            torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 2
            for i in [0, 1]
        ])
        D_mean_loss = sum([
            torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 2
            for i in [2, 3]
        ])
        validity = sum([
            torch.stack([x['validity'] for x in outputs[i]]).mean().item() / 2
            for i in [0, 1]
        ])
        reconstr = sum([
            torch.stack([x['reconstr'] for x in outputs[i]]).mean().item() / 2
            for i in [0, 1]
        ])

        identity = sum([
            torch.stack([x['identity'] for x in outputs[i]]).mean().item() / 2
            for i in [0, 1]
        ])

        self.losses.append(avg_loss)
        self.G_mean_losses.append(G_mean_loss)
        self.D_mean_losses.append(D_mean_loss)
        self.validity.append(validity)
        self.reconstr.append(reconstr)
        self.identity.append(identity)

        if self.step % 30 == 0:
            # Display Model Output
            target_img_paths = glob.glob(
                './gan-getting-started/photo_jpg/*.jpg')[:4]
            target_imgs = [
                self.transform(Image.open(path), phase='test')
                for path in target_img_paths
            ]
            target_imgs = torch.stack(target_imgs, dim=0)
            target_imgs = target_imgs.cuda()

            gen_imgs = self.G_basestyle(target_imgs)
            gen_img = torch.cat([target_imgs, gen_imgs], dim=0)

            # Reverse Normalization
            gen_img = gen_img * 0.5 + 0.5
            gen_img = gen_img * 255

            joined_images_tensor = make_grid(gen_img, nrow=4, padding=2)

            joined_images = joined_images_tensor.detach().cpu().numpy().astype(
                int)
            joined_images = np.transpose(joined_images, [1, 2, 0])

            # Visualize
            fig = plt.figure(figsize=(18, 8))
            plt.imshow(joined_images)
            plt.axis('off')
            plt.title(f'Epoch {self.step}')
            plt.savefig(f'./images/{self.step}.png')

        return None


# %%
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d'
        ) != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


# %% [markdown]
# ---
# ## Train

# %%
# Config  -----------------------------------------------------------------
data_dir = './gan-getting-started'
transform = ImageTransform(img_size=256)
batch_size = 1
lr = {'G': 0.0002, 'D': 0.0002}
epoch = 180
seed = 42
reconstr_w = 10
id_w = 2
seed_everything(seed)

# DataModule  -----------------------------------------------------------------
dm = MonetDataModule(data_dir, transform, batch_size, seed=seed)

G_basestyle = CycleGAN_Unet_Generator()
G_stylebase = CycleGAN_Unet_Generator()
D_base = CycleGAN_Discriminator()
D_style = CycleGAN_Discriminator()

# Init Weight  --------------------------------------------------------------
for net in [G_basestyle, G_stylebase, D_base, D_style]:
    init_weights(net, init_type='normal')

# LightningModule  --------------------------------------------------------------
model = CycleGAN_LightningSystem(G_basestyle, G_stylebase, D_base, D_style, lr,
                                 transform, reconstr_w, id_w)

# Trainer  --------------------------------------------------------------
trainer = Trainer(
    logger=False,
    max_epochs=epoch,
    gpus=1,
    enable_checkpointing=False,
    reload_dataloaders_every_n_epochs=True,
    num_sanity_val_steps=0,  # Skip Sanity Check
)

# Train
trainer.fit(model, datamodule=dm)

# Loss Plot
fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(18, 12), facecolor='w')
epoch_num = len(model.losses)

axes[0].plot(np.arange(epoch_num), model.losses, label='overall')
axes[0].plot(np.arange(epoch_num), model.G_mean_losses, label='generator')
axes[0].plot(np.arange(epoch_num), model.D_mean_losses, label='discriminator')
axes[0].legend()
axes[0].set_xlabel('Epoch')

axes[1].plot(np.arange(epoch_num), model.validity, label='validity')
axes[1].plot(np.arange(epoch_num), model.reconstr, label='reconstr')
axes[1].plot(np.arange(epoch_num), model.identity, label='identity')
axes[1].legend()
axes[1].set_xlabel('Epoch')

plt.show()

# %%
