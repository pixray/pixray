# Originally made by Katherine Crowson (https://github.com/crowsonkb, https://twitter.com/RiversHaveWings)
# The original BigGAN+CLIP method was by https://twitter.com/advadnoun

from DrawingInterface import DrawingInterface

import sys
import subprocess
sys.path.append('taming-transformers')
import os.path
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan

vqgan_config_table = {
    "imagenet_f16_1024": 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.yaml',
    "imagenet_f16_16384": 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1',
    "imagenet_f16_16384m": 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.yaml',
    "openimages_f16_8192": 'https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1',
    "coco": 'https://dl.nmkd.de/ai/clip/coco/coco.yaml',
    "faceshq": 'https://drive.google.com/uc?export=download&id=1fHwGx_hnBtC8nsq7hesJvs-Klv-P0gzT',
    "wikiart_1024": 'http://mirror.io.community/blob/vqgan/wikiart.yaml',
    "wikiart_16384": 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.yaml',
    "wikiart_16384m": 'http://mirror.io.community/blob/vqgan/wikiart_16384.yaml',
    "sflckr": 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fconfigs%2F2020-11-09T13-31-51-project.yaml&dl=1',
}
vqgan_checkpoint_table = {
    "imagenet_f16_1024": 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_1024.ckpt',
    "imagenet_f16_16384": 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1',
    "imagenet_f16_16384m": 'http://mirror.io.community/blob/vqgan/vqgan_imagenet_f16_16384.ckpt',
    "openimages_f16_8192": 'https://heibox.uni-heidelberg.de/d/2e5662443a6b4307b470/files/?p=%2Fckpts%2Flast.ckpt&dl=1',
    "coco": 'https://dl.nmkd.de/ai/clip/coco/coco.ckpt',
    "faceshq": 'https://app.koofr.net/content/links/a04deec9-0c59-4673-8b37-3d696fe63a5d/files/get/last.ckpt?path=%2F2020-11-13T21-41-45_faceshq_transformer%2Fcheckpoints%2Flast.ckpt',
    "wikiart_1024": 'http://mirror.io.community/blob/vqgan/wikiart.ckpt',
    "wikiart_16384": 'http://eaidata.bmk.sh/data/Wikiart_16384/wikiart_f16_16384_8145600.ckpt',
    "wikiart_16384m": 'http://mirror.io.community/blob/vqgan/wikiart_16384.ckpt',
    "sflckr": 'https://heibox.uni-heidelberg.de/d/73487ab6e5314cb5adba/files/?p=%2Fcheckpoints%2Flast.ckpt&dl=1'
}

def wget_file(url, out):
    try:
        output = subprocess.check_output(['wget', '-O', out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)

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

class VqganDrawer(DrawingInterface):
    def __init__(self, vqgan_model):
        super(DrawingInterface, self).__init__()
        self.vqgan_model = vqgan_model

    def load_model(self, config_path, checkpoint_path, device):
        gumbel = False

        if config_path is None:
            config_path = f'models/vqgan_{self.vqgan_model}.yaml'

        if checkpoint_path is None:
            checkpoint_path = f'models/vqgan_{self.vqgan_model}.ckpt'

        if not os.path.exists(config_path):
            wget_file(vqgan_config_table[self.vqgan_model], config_path)
        if not os.path.exists(checkpoint_path):
            wget_file(vqgan_checkpoint_table[self.vqgan_model], checkpoint_path)

        config = OmegaConf.load(config_path)
        if config.model.target == 'taming.models.vqgan.VQModel':
            model = vqgan.VQModel(**config.model.params)
            model.eval().requires_grad_(False)
            model.init_from_ckpt(checkpoint_path)
        elif config.model.target == 'taming.models.vqgan.GumbelVQ':
            model = vqgan.GumbelVQ(**config.model.params)
            model.eval().requires_grad_(False)
            model.init_from_ckpt(checkpoint_path)
            gumbel = True
        elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
            parent_model = cond_transformer.Net2NetTransformer(**config.model.params)
            parent_model.eval().requires_grad_(False)
            parent_model.init_from_ckpt(checkpoint_path)
            model = parent_model.first_stage_model
        else:
            raise ValueError(f'unknown model type: {config.model.target}')
        del model.loss

        # model, gumbel = load_vqgan_model(vqgan_config, vqgan_checkpoint)
        self.model = model.to(device)
        self.gumbel = gumbel
        self.device = device

        if gumbel:
            self.e_dim = 256
            self.n_toks = model.quantize.n_embed
            self.z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
            self.z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
        else:
            self.e_dim = model.quantize.e_dim
            self.n_toks = model.quantize.n_e
            self.z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
            self.z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

    def get_opts(self, decay_divisor):
        return None

    def rand_init(self, toksX, toksY):
        # legacy init
        one_hot = F.one_hot(torch.randint(self.n_toks, [toksY * toksX], device=self.device), n_toks).float()
        if self.gumbel:
            self.z = one_hot @ self.model.quantize.embed.weight
        else:
            self.z = one_hot @ self.model.quantize.embedding.weight

        self.z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        self.z.requires_grad_(True)

    def init_from_tensor(self, init_tensor):
        self.z, *_ = self.model.encode(init_tensor)        
        self.z.requires_grad_(True)

    def reapply_from_tensor(self, new_tensor):
        new_z, *_ = self.model.encode(new_tensor)        
        with torch.no_grad():
            self.z.copy_(new_z)

    def get_z_from_tensor(self, ref_tensor):
        z_ref, *_ = self.model.encode(ref_tensor)
        return z_ref

    def get_num_resolutions(self):
        return self.model.decoder.num_resolutions

    def synth(self, cur_iteration):
        if self.gumbel:
            z_q = vector_quantize(self.z.movedim(1, 3), self.model.quantize.embed.weight).movedim(3, 1)       # Vector quantize
        else:
            z_q = vector_quantize(self.z.movedim(1, 3), self.model.quantize.embedding.weight).movedim(3, 1)
        return clamp_with_grad(self.model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def to_image(self):
        out = self.synth(None)
        return TF.to_pil_image(out[0].cpu())

    def clip_z(self):
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

    def get_z(self):
        return self.z

    def set_z(self, new_z):
        with torch.no_grad():
            return self.z.copy_(new_z)

    def get_z_copy(self):
        return self.z.clone()
        # return model, gumbel

### EXTERNAL INTERFACE
### load_vqgan_model

if __name__ == '__main__':
    main()
