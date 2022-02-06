from filters.FilterInterface import FilterInterface
import torch

import kornia
import kornia.augmentation as K
import torch.nn as nn

from util import str2bool

class TilerFilter(FilterInterface):
    """
    Random tiled shifts in x and y with no loss
    """
    def __init__(self, settings, device):
        super().__init__(settings, device)

    def forward(self, imgs):
        loss = torch.tensor(0, device=self.device)
        B, C, H, W = imgs.size()
        rand_w = torch.randint(0, W, (1,))
        rand_h = torch.randint(0, H, (1,))
        imgs = torch.roll(imgs, shifts=(rand_h, rand_w), dims=(2, 3))
        return imgs, loss

