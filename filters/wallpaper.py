from filters.FilterInterface import FilterInterface
import torch

import kornia
import kornia.augmentation as K
import torch.nn as nn

from util import str2bool

class WallpaperFilter(FilterInterface):
    """
    Random tiled shifts in x and y with no loss
    """
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--wallpaper_shift", type=str2bool, help="shift offset rows", default=False, dest='wallpaper_shift')
        return parser

    def __init__(self, settings, device):
        super().__init__(settings, device)
        self.wallpaper_shift = settings.wallpaper_shift

    def forward(self, imgs):
        B, C, H, W = imgs.size()
        rand_w = torch.randint(0, W, (1,))
        rand_h = torch.randint(0, H, (1,))
        if self.wallpaper_shift:
            # rand_w = int(W/2)
            # rand_h = int(H/2)
            half_W = int(W / 2)
            row1 = imgs.tile((1,1,1,2))
            row2 = imgs.tile((1,1,1,2))
            row2 = torch.roll(row2, shifts=(half_W,), dims=(3,))
            two_rows = torch.cat([row1, row2], dim=2)
            # print(f"Went from {imgs.shape} to {two_rows.shape}")
            imgs = two_rows[:,:,rand_h:(rand_h+H),rand_w:(rand_w+W)]
        else:
            imgs = torch.roll(imgs, shifts=(rand_h, rand_w), dims=(2, 3))
        return imgs, torch.tensor(0, device=self.device)

