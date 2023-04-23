import torch
from torch import nn
import torch.nn.init as init

from PIL import Image
from torchvision.transforms.functional import to_tensor, resize

from utils import SpectralNorm


class Self_Attn(nn.Module):
    def __init__(self, in_features):
        super(Self_Attn, self).__init__()

        self.f_conv = nn.Conv2d(in_features, in_features, kernel_size=(1, 1))
        self.g_conv = nn.Conv2d(in_features, in_features, kernel_size=(1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, _, w, h = x.size()

        f = self.f_conv(x).view(bs, -1, w*h).permute(0, 2, 1) # B X W*H X C
        g = self.g_conv(x).view(bs, -1, w*h) # B X C X W*H
        energy = torch.bmm(f, g)
        attn = self.sm(energy)

        out = torch.bmm(x.view(bs, -1, w*h), attn.permute(0, 2, 1))
        out = out.view(bs, -1, w, h)

        out = self.gamma*out + x

        return out
    

class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn1 = Self_Attn(cfg.generator_hdim*2)
        self.attn2 = Self_Attn(cfg.generator_hdim)

        self.l1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(cfg.nz, cfg.generator_hdim*8, kernel_size=(4, 4), stride=(2, 2), bias=False)),
            nn.BatchNorm2d(cfg.generator_hdim*8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.l2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(cfg.generator_hdim*8, cfg.generator_hdim*4, kernel_size=(4, 4), stride=(4, 4), bias=False)),
            nn.BatchNorm2d(cfg.generator_hdim*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg.g_dropout)
        )

        self.l3 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(cfg.generator_hdim*4, cfg.generator_hdim*2, kernel_size=(2, 2), stride=(2, 2), bias=False)),
            nn.BatchNorm2d(cfg.generator_hdim*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg.g_dropout)
        )

        self.l4 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(cfg.generator_hdim*2, cfg.generator_hdim, kernel_size=(2, 2), stride=(2, 2), bias=False)),
            nn.BatchNorm2d(cfg.generator_hdim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg.g_dropout)
        )
        
        self.l5 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(cfg.generator_hdim, cfg.channels_number, kernel_size=(2, 2), stride=(2, 2), bias=False)),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.l1(x) # [bs, 64, 4, 4]
        x = self.l2(x) # [bs, 32, 16, 16]
        x = self.attn1(self.l3(x)) # [bs, 16, 32, 32]
        x = self.attn2(self.l4(x)) # [bs, 3, 64, 64]
        out = self.l5(x)

        return out


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        ndf = cfg.discriminator_hdim

        self.attn1 = Self_Attn(ndf*2)
        self.attn2 = Self_Attn(ndf*4)        

        self.l1 = nn.Sequential(

            SpectralNorm(nn.Conv2d(cfg.channels_number, ndf, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1), bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg.d_dropout)
            
        )
            
        self.l2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=(4, 4), stride=(3, 3), padding=(1, 1), bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg.d_dropout)
        )

        self.l3 = nn.Sequential(

            SpectralNorm(nn.Conv2d(ndf * 4, 1, kernel_size=(5, 5), stride=(4, 4), padding=(1, 1), bias=False)),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.attn1(self.l1(x))
        x = self.attn2(self.l2(x))
        out = self.l3(x)

        return out
       

