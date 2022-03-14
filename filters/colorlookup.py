from filters.FilterInterface import FilterInterface

import torch
from torch import nn, optim
from torch.nn import functional as F

from einops import rearrange

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

class ColorLookup(FilterInterface):
    """
    Maps to fixed color table
    """
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--lookup_beta", type=float, help="loss scaling", default=10.0, dest='lookup_beta')
        return parser

    def __init__(self, settings, device):
        super().__init__(settings, device)

        self.beta = settings.lookup_beta
        color_table = settings.palette

        if color_table is None:
            print("WARNING: using built in palette")
            # eventually no table would mean make up your own table?
            color_table = default_color_table

        print(f"color table has {len(color_table)} entries like {color_table[0:5]}")
        self.color_table = torch.FloatTensor(color_table).to(device)

    # https://discuss.pytorch.org/t/how-to-find-k-nearest-neighbor-of-a-tensor/51593
    def forward(self, z):
        B, C, H, W = z.size()
        # print("z coming in is ", z.size())

        do_alpha = False
        if C == 4:
            alpha = z[:,3,:,:]
            z3 = z[:,0:3,:,:]
            do_alpha = True
        else:
            z3 = z

        # project and flatten out space, so (B, C, H, W) -> (B*H*W, C)
        # reshape z -> (batch, height, width, channel) and flatten
        z3 = rearrange(z3, 'b c h w -> b h w c').contiguous()

        ind = torch.cdist(z3, self.color_table).argmin(axis=-1)
        z_q = torch.index_select(self.color_table, 0, ind.flatten()).view(z3.shape)

        loss = self.beta * torch.mean((z_q.detach()-z3)**2) + \
               torch.mean((z_q - z3.detach()) ** 2)

        # preserve gradients
        z_q = z3 + (z_q - z3).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        if do_alpha:
            # z_q = torch.stack([z_q, alpha], dim=1)
            z[:,0:3,:,:] = z_q
            z[:,3,:,:] = alpha
            # print("z_q with alpha is now ", z_q.shape)
        else:
            z = z_q
        return z, loss

