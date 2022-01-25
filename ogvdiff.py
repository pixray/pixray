# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

from DrawingInterface import DrawingInterface

import sys
import subprocess
sys.path.append('v-diffusion-pytorch')
import os
import os.path
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF
import math
from torchvision.utils import save_image

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan


model_urls = {
    "yfcc_2":"https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_2.pth",
    "yfcc_1":"https://v-diffusion.s3.us-west-2.amazonaws.com/yfcc_1.pth",
    "cc12m_1":"https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1.pth",
    "cc12m_1_cfg":"https://v-diffusion.s3.us-west-2.amazonaws.com/cc12m_1_cfg.pth",
}


def wget_file(url, out):
    try:
        print(f"Downloading {out} from {url}, please wait")
        output = subprocess.check_output(['wget', '-O', out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)

from pathlib import Path
MODULE_DIR = Path(__file__).resolve().parent

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

class OGVdiffDrawer(DrawingInterface):
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--vdiff_model", type=str, help="VDIFF model from [yfcc_2, yfcc_1, cc12m_1,cc12m_1_cfg]", default='yfcc_2', dest='vdiff_model')
        parser.add_argument("--vdiff_init_skip", type=float, help="skip steps (step power) when init", default=0.9, dest='vdiff_init_skip')
        # parser.add_argument("--vqgan_config", type=str, help="VQGAN config", default=None, dest='vqgan_config')
        # parser.add_argument("--vqgan_checkpoint", type=str, help="VQGAN checkpoint", default=None, dest='vqgan_checkpoint')
        return parser

    def __init__(self, settings):
        super(DrawingInterface, self).__init__()
        assert (settings.vdiff_model != "cc12m_1" and settings.vdiff_model != "cc12m_1_cfg")or ( "RN50x4" not in settings.clip_models and  "RN50" not in settings.clip_models), "try clip_models='RN101,ViT-B/32,ViT-B/16' or only RN50 or only RN50x4, the clip embedding of RN50 and RN50x4 is not suitable with the rest"
        os.makedirs("models",exist_ok=True)
        self.vdiff_model = settings.vdiff_model
        self.canvas_width = settings.size[0]
        self.canvas_height = settings.size[1]
        self.gen_width = roundup(self.canvas_width, ROUNDUP_SIZE)
        self.gen_height = roundup(self.canvas_height, ROUNDUP_SIZE)
        self.iterations = settings.iterations
        self.eta = 1.0
        self.init_image = settings.init_image
        self.vdiff_init_skip = settings.vdiff_init_skip
        self.total_its = settings.iterations

    def load_model(self, settings, device):
        model = get_model(self.vdiff_model)()
        # checkpoint = MODULE_DIR / f'checkpoints/{self.vdiff_model}.pth'
        checkpoint = f'models/{self.vdiff_model}.pth'
        
        if not (os.path.exists(checkpoint) and os.path.isfile(checkpoint)):
            wget_file(model_urls[self.vdiff_model],checkpoint)

        model.load_state_dict(torch.load(checkpoint, map_location='cpu'))
        if device.type == 'cuda':
            model = model.half()
        model = model.to(device).eval().requires_grad_(False)

        self.model = model
        self.device = device
        self.pred = None
        self.v = None
        # self.x = torch.randn([1, 3, self.gen_height, self.gen_width], device=self.device)
        # self.x.requires_grad_(True)
        # self.t = torch.linspace(1, 0, self.iterations+2, device=self.device)[:-1]
        # self.steps = utils.get_spliced_ddpm_cosine_schedule(self.t)
        # # [model, steps, eta, extra_args, ts, alphas, sigmas]
        # self.sample_state = sampling.sample_setup(self.model, self.x, self.steps, self.eta, {})


    def get_opts(self, decay_divisor):
        return None

    def rand_init(self, toksX, toksY):
        # legacy init
        return None

    def init_from_tensor(self, init_tensor):
        self.x = torch.randn([1, 3, self.gen_height, self.gen_width], device=self.device)
        self.t = torch.linspace(1, 0, self.iterations+2, device=self.device)[:-1]
        self.steps = utils.get_spliced_ddpm_cosine_schedule(self.t)
        # [model, steps, eta, extra_args, ts, alphas, sigmas]
        self.sample_state = sampling.sample_setup(self.model, self.x, self.steps, self.eta, {})

        
        if self.init_image is not None:
            self.steps = self.steps[self.steps < self.vdiff_init_skip]
            alpha, sigma = utils.t_to_alpha_sigma(self.steps)
            new_init_tensor = torch.zeros([1, 3, self.gen_height, self.gen_width], device=self.device)
            margin_x = int((self.gen_width - self.canvas_width)/2)
            margin_y = int((self.gen_height - self.canvas_height)/2)
            if (margin_x != 0 or margin_y != 0):
                new_init_tensor[:,:,margin_y:(margin_y+self.canvas_height),margin_x:(margin_x+self.canvas_width)] = init_tensor
            else:
                new_init_tensor = init_tensor
            self.x = new_init_tensor * alpha[0] + self.x * sigma[0]
            self.sample_state[5], self.sample_state[6] = alpha, sigma
            self.sample_state[1] = self.steps
            self.total_its = len(self.steps)-1
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

        # resize
        # pixels = TF.resize(pixels, (self.canvas_height, self.canvas_width))

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
