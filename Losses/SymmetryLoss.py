import torch
from torch import nn
from Losses.LossInterface import LossInterface


class SymmetryLoss(LossInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def add_settings(parser):
        parser.add_argument(
            "--symmetry_weight",
            type=float,
            help="how much symmetry is weighted in loss",
            default=1,
            dest='symmetry_weight')
        return parser

    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        mseloss = nn.MSELoss()
        cur_loss = mseloss(out, torch.flip(out, [3]))
        return cur_loss * args.symmetry_weight
