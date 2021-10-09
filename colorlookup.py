import argparse
import math
from urllib.request import urlopen
import sys
import os
import subprocess
import glob
from braceexpand import braceexpand
from types import SimpleNamespace

import os.path

from omegaconf import OmegaConf

import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
torch.backends.cudnn.benchmark = False		# NR: True is a bit faster, but can lead to OOM. False is more deterministic.
#torch.use_deterministic_algorithms(True)		# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation

from torch_optimizer import DiffGrad, AdamP, RAdam
from perlin_numpy import generate_fractal_noise_2d

# todo: fix this mess
try:
    # installed by adding github.com/openai/CLIP to sys.path
    from CLIP import clip
except ImportError:
    # installed by doing `pip install git+https://github.com/openai/CLIP`
    from clip import clip
import kornia
import kornia.augmentation as K
import numpy as np
import imageio
import random

from einops import rearrange

from PIL import ImageFile, Image, PngImagePlugin

default_color_table = []
default_color_table.append([  0/255.0,   0/255.0,   0/255.0])
default_color_table.append([255/255.0, 255/255.0, 255/255.0])
default_color_table.append([ 63/255.0,  40/255.0,  50/255.0])
default_color_table.append([ 38/255.0,  43/255.0,  68/255.0])
default_color_table.append([ 90/255.0, 105/255.0, 136/255.0])
default_color_table.append([139/255.0, 155/255.0, 180/255.0])
default_color_table.append([ 25/255.0,  60/255.0,  62/255.0])
default_color_table.append([ 38/255.0,  92/255.0,  66/255.0])
default_color_table.append([ 62/255.0, 137/255.0,  72/255.0])
default_color_table.append([ 99/255.0, 199/255.0,  77/255.0])
default_color_table.append([254/255.0, 231/255.0,  97/255.0])
default_color_table.append([254/255.0, 174/255.0,  52/255.0])
default_color_table.append([254/255.0, 174/255.0,  52/255.0])
default_color_table.append([247/255.0, 118/255.0,  34/255.0])
default_color_table.append([184/255.0, 111/255.0,  80/255.0])
default_color_table.append([116/255.0,  63/255.0,  57/255.0])

from scipy.cluster.vq import kmeans2

class ColorLookup(nn.Module):
    """
    Maps to fixed color table
    """
    def __init__(self, color_table, device, beta=10.0):
        super().__init__()

        self.beta = beta

        if color_table is None:
            print("WARNING: using built in palette")
            # eventually no table would mean make up your own table?
            color_table = default_color_table

        print(f"color table has {len(color_table)} entries like {color_table[0:5]}")
        self.color_table = torch.FloatTensor(color_table).to(device)

    # https://discuss.pytorch.org/t/how-to-find-k-nearest-neighbor-of-a-tensor/51593
    def forward(self, z):
        B, C, H, W = z.size()

        # project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()

        ind = torch.cdist(z, self.color_table).argmin(axis=-1)
        z_q = torch.index_select(self.color_table, 0, ind.flatten()).view(z.shape)

        loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
               torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        return z_q, loss

