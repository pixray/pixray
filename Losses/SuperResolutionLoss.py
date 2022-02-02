import os
import torch
from torch import nn, optim
import torch.nn.functional as F
from Losses.LossInterface import LossInterface
from basicsr.archs.rrdbnet_arch import RRDBNet

from real_esrganer import RealESRGANer

from util import wget_file

superresolution_checkpoint_table = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


class SuperResolutionLoss(LossInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        print("SuperResolutionLoss", kwargs)
        self.super_resolution_model = "RealESRGAN_x4plus"

        checkpoint_path = f'models/super_resolution_{self.super_resolution_model}.ckpt'
        if not os.path.exists(checkpoint_path):
            wget_file(superresolution_checkpoint_table[self.super_resolution_model], checkpoint_path)

        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

        self.upsampler = RealESRGANer(
            scale=4,
            model_path=checkpoint_path,
            model=self.model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
        )
    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--super_resolution_model", type=str, help="Super resolution model", default="RealESRGAN_x4plus", dest="super_resolution_model")
        return parser

    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        with torch.no_grad():
            downsampled_shape = torch.tensor(out.shape[-2:]) // 4
            upsampled_shape = downsampled_shape * 4
            downsampled = F.interpolate(out, size=downsampled_shape.tolist(), mode="bilinear", align_corners=False)
            upsampled = self.upsampler.enhance(downsampled, outscale=4)
            upsampled = F.interpolate(upsampled, size=upsampled_shape.tolist(), mode="bilinear", align_corners=False)
        return torch.norm(out - upsampled, p=2, dim=1).mean() * 0.001
