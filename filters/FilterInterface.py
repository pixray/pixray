import argparse
from torch import nn

class FilterInterface(nn.Module):
    @staticmethod
    def add_settings(parser):
        #add parser.add_argument() here
        return parser

    def __init__(self, settings, device=None):
        super().__init__()
        self.device = device

    # implement forward(img) and return img, loss
    def forward(self, img):
        return img, 0
