
import torch
from torch import nn, optim

from util import  palette_from_string



from Losses.LossInterface import LossInterface

class PaletteLoss(LossInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--palette_weight", type=float, help="strength of pallete loss effect", default=1, dest='palette_weight')
        return parser

    def parse_settings(self,args):
        #do stuff with args here
        # args.target_palette = palette_from_string(args.target_palette)
        return args
    
    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        target_palette = torch.FloatTensor(args.palette).requires_grad_(False).to(self.device)
        all_loss = []
        for _,cutouts in cur_cutouts.items():
            _pixels = cutouts.permute(0,2,3,1).reshape(-1,3)
            palette_dists = torch.cdist(target_palette, _pixels, p=2)
            best_guesses = palette_dists.argmin(axis=0)
            diffs = _pixels - target_palette[best_guesses]
            palette_loss = torch.mean( torch.norm( diffs, 2, dim=1 ) )*cutouts.shape[0]
            all_loss.append( palette_loss * args.palette_weight/10.0 )
        return all_loss
