import torch
import sys
import os
from torch import nn, optim
from Losses.LossInterface import LossInterface
import resmem
from resmem import ResMem, transformer
from torchvision import transforms

recenter = transforms.Compose((
    transforms.Resize((256, 256)),
    transforms.CenterCrop(227),
    )
)

from util import wget_file, map_number
resmem_url = 'https://github.com/pixray/resmem/releases/download/1.1.3_model/model.pt'

class ResmemLoss(LossInterface):
    def __init__(self,**kwargs):
        # make sure resmem has model file
        if not os.path.exists(resmem.path):
            wget_file(resmem_url, resmem.path)        

        # TODO: device should be part of init
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        super().__init__(**kwargs)
        self.model = ResMem(pretrained=True).to(self.device)
        # Set the model to inference mode.
        self.model.eval()

    @staticmethod
    def add_settings(parser):
        parser.add_argument("--symmetry_weight", type=float, help="how much symmetry is weighted in loss", default=1, dest='symmetry_weight')
        return parser
   
    def get_loss1(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        device = self.device

        # print(out)
        # print(out.shape)
        image_x = recenter(out)
        # print(image_x.shape)

        prediction = self.model(image_x.view(-1, 3, 227, 227))
        # print(prediction)
        # print(prediction.shape)
        the_loss = 1.0 - prediction[0][0]

        return the_loss

    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        device = self.device

        # print(cur_cutouts.keys())
        images = cur_cutouts[224]
        # print(images.shape)
        image_x = recenter(images)
        # print(image_x.shape)

        prediction = self.model(image_x)
        # print(prediction)
        # print(prediction.shape)
        mean = torch.mean(prediction)
        # loss seems to bottom out at 0.4? ¯\_(ツ)_/¯
        mapped_mean = map_number(mean, 0.4, 1.0, 0, 1)
        the_loss = 0.05 * mapped_mean

        return the_loss
