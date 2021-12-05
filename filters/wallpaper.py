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
        parser.add_argument("--wallpaper_type", type=str, help="none, shift, horizontal", default=None, dest='wallpaper_type')
        parser.add_argument("--wallpaper_edge_match", type=int, help="force repeating match in pixels", default=0, dest='wallpaper_edge_match')
        return parser

    def __init__(self, settings, device):
        super().__init__(settings, device)
        self.wallpaper_type = settings.wallpaper_type
        self.edge_match = settings.wallpaper_edge_match

    def forward(self, imgs):
        loss = torch.tensor(0, device=self.device)
        B, C, H, W = imgs.size()
        rand_w = torch.randint(0, W, (1,))
        rand_h = torch.randint(0, H, (1,))
        if self.wallpaper_type == "shift":
            # rand_w = int(W/2)
            # rand_h = int(H/2)
            half_W = int(W / 2)
            row1 = imgs.tile((1,1,1,1))
            row2 = imgs.tile((1,1,1,1))
            row2 = torch.roll(row2, shifts=(half_W,), dims=(3,))
            two_rows = torch.cat([row1, row2], dim=2)
            # print(f"Went from {imgs.shape} to {two_rows.shape}")
            # imgs = two_rows[:,:,rand_h:(rand_h+H),rand_w:(rand_w+W)]
            # imgs = two_rows[:,:,:,rand_w:(rand_w+W)]
            imgs = torch.roll(two_rows, shifts=(rand_h, rand_w), dims=(2, 3))
        elif self.wallpaper_type == "horizontal":
            if self.edge_match != 0:
                em = self.edge_match
                em2 = int(em / 2)
                # first trim edge and compute loss
                col1 = imgs[:,:,:,:em]
                col2 = imgs[:,:,:, -em:]
                mseloss = nn.MSELoss()
                loss = mseloss(col1, col2) * 100 / em
                # print(col1.shape, col2.shape, imgs.shape, em, em2)
                imgs = imgs[:,:,:,em2:-em2]
                # print(col1.shape, col2.shape, imgs.shape, em, em2)
            imgs = torch.roll(imgs, shifts=(rand_w,), dims=(3,))
        else:
            imgs = torch.roll(imgs, shifts=(rand_h, rand_w), dims=(2, 3))
        return imgs, loss

