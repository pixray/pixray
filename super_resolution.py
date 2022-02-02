# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

from mimetypes import init
from DrawingInterface import DrawingInterface

import sys
import subprocess
sys.path.append('taming-transformers')
import os.path
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
from basicsr.archs.rrdbnet_arch import RRDBNet

from real_esrganer import RealESRGANer


from util import wget_file

superresolution_checkpoint_table = {
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply

def vector_quantize(x, codebook):
    d = x.pow(2).sum(dim=-1, keepdim=True) + codebook.pow(2).sum(dim=1) - 2 * x @ codebook.T
    indices = d.argmin(-1)
    x_q = F.one_hot(indices, codebook.shape[0]).to(d.dtype) @ codebook
    return replace_grad(x_q, x)

class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        ctx.min = min
        ctx.max = max
        ctx.save_for_backward(input)
        return input.clamp(min, max)

    @staticmethod
    def backward(ctx, grad_in):
        input, = ctx.saved_tensors
        return grad_in * (grad_in * (input - input.clamp(ctx.min, ctx.max)) >= 0), None, None

clamp_with_grad = ClampWithGrad.apply

global_model_cache = {}

class SuperResolutionDrawer(DrawingInterface):
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--super_resolution_model", type=str, help="Super resolution model", default="RealESRGAN_x4plus", dest="super_resolution_model")
        return parser

    def __init__(self, settings):
        super(DrawingInterface, self).__init__()
        self.super_resolution_model = settings.super_resolution_model

    def load_model(self, settings, device):
        global global_model_cache

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

    def get_opts(self, decay_divisor):
        return None

    def init_from_tensor(self, init_tensor):
        self.z = self.get_z_from_tensor(init_tensor)
        self.z.requires_grad_(True)

    def reapply_from_tensor(self, new_tensor):
        new_z = self.get_z_from_tensor(new_tensor)
        with torch.no_grad():
            self.z.copy_(new_z)

    def get_z_from_tensor(self, ref_tensor):
        return F.interpolate((ref_tensor + 1) / 2, size=(torch.tensor(ref_tensor.shape[-2:]) // 4).tolist(), mode="bilinear", align_corners=False)

    def get_num_resolutions(self):
        return None

    def synth(self, cur_iteration):
        output = self.upsampler.enhance(self.z, outscale=4)
        return clamp_with_grad(output, 0, 1)

    @torch.no_grad()
    def to_image(self):
        out = self.synth(None)
        return TF.to_pil_image(out[0].cpu())

    def clip_z(self):
        with torch.no_grad():
            self.z.copy_(self.z.clip(0, 1))

    def get_z(self):
        return self.z

    def set_z(self, new_z):
        with torch.no_grad():
            return self.z.copy_(new_z)

    def get_z_copy(self):
        return self.z.clone()
        # return model, gumbel
