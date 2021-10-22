
import torch
from torch import nn, optim

from util import  palette_from_string



from LossInterface import LossInterface

class PaletteLoss(LossInterface):
    def __init__(self,device):
        self.device = device
        super().__init__()
    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("-epw",  "--enforce_palette_annealing", type=int, help="enforce palette annealing, 0 -- skip", default=5000, dest='enforce_palette_annealing')
        parser.add_argument("-tp",   "--target_palette", type=str, help="target palette", default=None, dest='target_palette')
        parser.add_argument("-tpl",  "--target_palette_length", type=int, help="target palette length", default=16, dest='target_palette_length')
        return parser
    


    def parse_settings(self,args):
        #do stuff with args here
        args.target_palette = palette_from_string(args.target_palette)
        return args
    
    def add_globals(self,args):
        globals = {}
        return globals
    
    def forward(self, cur_cutouts, out, args, globals):
        target_palette = torch.FloatTensor(args.target_palette).requires_grad_(False).to(self.device)
        all_loss = []
        for i,cutouts in cur_cutouts.items():
            _pixels = cutouts.permute(0,2,3,1).reshape(-1,3)
            palette_dists = torch.cdist(target_palette, _pixels, p=2)
            best_guesses = palette_dists.argmin(axis=0)
            diffs = _pixels - target_palette[best_guesses]
            palette_loss = torch.mean( torch.norm( diffs, 2, dim=1 ) )*cutouts.shape[0]
            all_loss.append( palette_loss*globals['cur_iteration']/args.enforce_palette_annealing )
        return all_loss
