# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

from DrawingInterface import DrawingInterface

import sys
import os
import subprocess

# TODO: this is very hacky, must fix this later (submodule dependency)
VDIFF_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'v-diffusion-pytorch')
sys.path.append(VDIFF_PATH)

import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
import math

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
from util import wget_file, map_number


model_urls = {
    "yfcc_2":      "https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_2.pth",
    "yfcc_1":      "https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_1.pth",
    "cc12m_1":     "https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1.pth",
    "cc12m_1_cfg": "https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1_cfg.pth"
}


from pathlib import Path

from diffusion import get_model, get_models, sampling, utils

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

ROUNDUP_SIZE = 128
# https://stackoverflow.com/a/8866125/1010653
def roundup(x, n):
    return int(math.ceil(x / float(n))) * n

class VdiffDrawer(DrawingInterface):
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--vdiff_model", type=str, help="VDIFF model from [yfcc_2, yfcc_1, cc12m_1, cc12m_1_cfg]", default='yfcc_2', dest='vdiff_model')
        parser.add_argument("--vdiff_schedule", type=str, help="VDIFF schedule [default, log]", default="default", dest='vdiff_schedule')
        parser.add_argument("--vdiff_skip", type=float, help="skip a percentage of the way into the decay schedule (0-100)", default=0, dest='vdiff_skip')
        return parser

    def __init__(self, settings):
        super(DrawingInterface, self).__init__()
        os.makedirs("models",exist_ok=True)
        self.vdiff_model = settings.vdiff_model
        self.canvas_width = settings.size[0]
        self.canvas_height = settings.size[1]
        self.gen_width = roundup(self.canvas_width, ROUNDUP_SIZE)
        self.gen_height = roundup(self.canvas_height, ROUNDUP_SIZE)
        self.iterations = settings.iterations
        self.schedule = settings.vdiff_schedule
        self.active_clip_models = settings.clip_models
        self.eta = 1
        self.vdiff_skip = settings.vdiff_skip

    def load_model(self, settings, device):
        model = get_model(self.vdiff_model)()
        checkpoint = f'models/{self.vdiff_model}.pth'
        
        if not (os.path.exists(checkpoint) and os.path.isfile(checkpoint)):
            wget_file(model_urls[self.vdiff_model],checkpoint)

        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        if device.type == 'cuda':
            model = model.half()
        model = model.to(device).eval().requires_grad_(False)

        if hasattr(model, 'clip_model'):
            self.clip_model = model.clip_model
            assert self.clip_model in self.active_clip_models, f"try adding {self.clip_model} to clip_models settings - vdiff model {self.vdiff_model} needs it but it is not active"
        else:
            self.clip_model = None

        self.model = model
        self.device = device
        self.pred = None
        self.v = None

    def get_opts(self, decay_divisor):
        return None

    def rand_init(self, toksX, toksY):
        # legacy init
        return None

    def init_from_tensor(self, init_tensor):

        # compute self.t based on vdiff_skip
        top_val = map_number(self.vdiff_skip, 0, 100, 1, 0)
        # print("Using a max for vdiff skip of ", top_val)
        self.t = torch.linspace(top_val, 0, self.iterations+2, device=self.device)[:-1]
        # print("self.t is ", self.t)

        self.x = torch.randn([1, 3, self.gen_height, self.gen_width], device=self.device)

        if self.schedule == 'log':
            self.steps = utils.get_log_schedule(self.t)
        else:
            self.steps = utils.get_spliced_ddpm_cosine_schedule(self.t)

        # todo: maybe scheduld should adjust better due to init_skip?
        if init_tensor is not None:
            # reverse-center crop
            new_x = torch.randn([1, 3, self.gen_height, self.gen_width], device=self.device)
            margin_x = int((self.gen_width - self.canvas_width)/2)
            margin_y = int((self.gen_height - self.canvas_height)/2)
            if (margin_x != 0 or margin_y != 0):
                new_x[:,:,margin_y:(margin_y+self.canvas_height),margin_x:(margin_x+self.canvas_width)] = init_tensor
            else:
                new_x = init_tensor
            # by default the image is 99% based on init_tensor (for now)
            self.x = new_x * 0.99 + self.x * 0.01

        # [model, steps, eta, extra_args, ts, alphas, sigmas]
        self.sample_state = sampling.sample_setup(self.model, self.x, self.steps, self.eta, {})
        self.x.requires_grad_(True)
        self.pred = None 
        self.v = None 

    def reapply_from_tensor(self, new_tensor):
        return None

    def get_z_from_tensor(self, ref_tensor):
        return None

    def get_num_resolutions(self):
        return None

    def makenoise(self, cur_it):
        return sampling.sample_noise(self.sample_state, self.x, cur_it, self.pred, self.v).detach()

    def synth(self, cur_iteration):
        pred, v, next_x = sampling.sample_step(self.sample_state, self.x, cur_iteration, self.pred, self.v)
        self.pred = pred.detach()
        self.v = v.detach()
        pixels = clamp_with_grad(pred.add(1).div(2), 0, 1)

        # center crop
        margin_x = int((self.gen_width - self.canvas_width)/2)
        margin_y = int((self.gen_height - self.canvas_height)/2)
        if (margin_x != 0 or margin_y != 0):
            pixels = pixels[:,:,margin_y:(margin_y+self.canvas_height),margin_x:(margin_x+self.canvas_width)]

        # save a copy for the next iteration
        return pixels

    @torch.no_grad()
    def to_image(self):
        out = self.synth(None)
        return TF.to_pil_image(out[0].cpu())

    def clip_z(self):
        return None

    def get_z(self):
        return self.x

    def set_z(self, new_z):
        with torch.no_grad():
            return self.x.copy_(new_z)

    def get_z_copy(self):
        return self.x.clone()
        # return model, gumbel
