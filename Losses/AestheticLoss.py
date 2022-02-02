import torch
from torch import nn, optim
from Losses.LossInterface import LossInterface
from util import wget_file
from torch.nn import functional as F


class AestheticLoss(LossInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        wget_file("https://dazhi.art/f/ava_vit_b_16_linear.pth", "models/ava_vit_b_16_linear.pth")
    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--aesthetic_target", type=float, help="0-10", default=10, dest='target_ava')
        return parser

    def parse_settings(self, args):

        layer_weights = torch.load("models/ava_vit_b_16_linear.pth")
        self.ae_reg = nn.Linear(512, 1).to(self.device)
        self.ae_reg.bias.data = layer_weights["bias"].to(self.device)
        self.ae_reg.weight.data = layer_weights["weight"].to(self.device)
        self.target_rating = torch.ones(size = (args.num_cuts, 1))*args.aesthetic_target
        self.target_rating =  self.target_rating.to(self.device)
        return args
   
    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        aes_rating = self.ae_reg(F.normalize(globals["embeds"], dim=-1)).to(self.device)
        aes_loss = (aes_rating-self.target_rating).square().mean() * 0.1
        return aes_loss
