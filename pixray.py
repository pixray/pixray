import argparse
import json
import math
import logging

from urllib.request import urlopen
import sys
import os
import subprocess
import json
import yaml
import glob
from braceexpand import braceexpand
from types import SimpleNamespace

import os.path

from omegaconf import OmegaConf
import hashlib

import time
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
torch.backends.cudnn.benchmark = False		# NR: True is a bit faster, but can lead to OOM. False is more deterministic.
#torch.use_deterministic_algorithms(True)		# NR: grid_sampler_2d_backward_cuda does not have a deterministic implementation

from torch_optimizer import DiffGrad, AdamP
from perlin_numpy import generate_fractal_noise_2d
from util import str2bool, get_file_path

from slip import get_clip_perceptor

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# installed by doing `pip install git+https://github.com/openai/CLIP`
from clip import clip

import kornia
import kornia.augmentation as K
import numpy as np
import imageio
import re
import random

from einops import rearrange

from filters.colorlookup import ColorLookup
from filters.wallpaper import WallpaperFilter

filters_class_table = {
    "lookup": ColorLookup,
    "wallpaper": WallpaperFilter,
}

from PIL import ImageFile, Image, PngImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True

# or 'border'
global_padding_mode = 'reflection'
global_aspect_width = 1
global_spot_file = None

from util import map_number, palette_from_string, real_glob

from vqgan import VqganDrawer
from vdiff import VdiffDrawer

class_table = {
    "vqgan": VqganDrawer,
    "vdiff": VdiffDrawer
}

try:
    from fftdrawer import FftDrawer
    # update class_table if these import OK
    class_table.update({"fft": FftDrawer})
except ImportError as e:
    print("--> Not running with fft support", e)
    pass

try:
    from clipdrawer import ClipDrawer
    from pixeldrawer import PixelDrawer
    from linedrawer import LineDrawer
    # update class_table if these import OK
    class_table.update({
        "line_sketch": LineDrawer,
        "pixel": PixelDrawer,
        "clipdraw": ClipDrawer
    })
except ImportError as e:
    print("--> Not running with pydiffvg drawer support ", e)
    pass

try:
    import matplotlib.colors
except ImportError:
    # only needed for palette stuff
    pass

from Losses.LossInterface import LossInterface
from Losses.PaletteLoss import PaletteLoss
from Losses.SaturationLoss import SaturationLoss
from Losses.SymmetryLoss import SymmetryLoss
from Losses.SmoothnessLoss import SmoothnessLoss
from Losses.EdgeLoss import EdgeLoss
from Losses.StyleLoss import StyleLoss
from Losses.ResmemLoss import ResmemLoss
from Losses.AestheticLoss import AestheticLoss

loss_class_table = {
    "palette": PaletteLoss,
    "saturation": SaturationLoss,
    "symmetry": SymmetryLoss,
    "smoothness": SmoothnessLoss,
    "edge": EdgeLoss,
    "style": StyleLoss,
    "resmem": ResmemLoss,
    "aesthetic": AestheticLoss,
}


# this is enabled when not in the master branch
# print("warning: running unreleased future version")

# https://stackoverflow.com/a/39662359
def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'Shell':
            return True   # Seems to be what co-lab does
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

IS_NOTEBOOK = isnotebook()

if IS_NOTEBOOK:
    from IPython import display
    from tqdm.notebook import tqdm
    from IPython.display import clear_output
else:
    from tqdm import tqdm

# Functions and classes
def sinc(x):
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x/a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


# NR: Testing with different intital images
def old_random_noise_image(w,h):
    random_image = Image.fromarray(np.random.randint(0,255,(w,h,3),dtype=np.dtype('uint8')))
    return random_image

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# https://stats.stackexchange.com/a/289477
def contrast_noise(n):
    n = 0.9998 * n + 0.0001
    n1 = (n / (1-n))
    n2 = np.power(n1, -2)
    n3 = 1 / (1 + n2)
    return n3

def random_noise_image(w,h):
    # scale up roughly as power of 2
    if (w>1024 or h>1024):
        side, octp = 2048, 6
    elif (w>512 or h>512):
        side, octp = 1024, 5
    elif (w>256 or h>256):
        side, octp = 512, 4
    else:
        side, octp = 256, 3

    nr = NormalizeData(generate_fractal_noise_2d((side, side), (32, 32), octp))
    ng = NormalizeData(generate_fractal_noise_2d((side, side), (32, 32), octp))
    nb = NormalizeData(generate_fractal_noise_2d((side, side), (32, 32), octp))
    stack = np.dstack((contrast_noise(nr),contrast_noise(ng),contrast_noise(nb)))
    substack = stack[:h, :w, :]
    im = Image.fromarray((255.999 * substack).astype('uint8'))
    return im

# testing
def gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=float)

    for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
        result[:, :, i] = gradient_2d(start, stop, width, height, is_horizontal)

    return result

    
def random_gradient_image(w,h):
    array = gradient_3d(w, h, (0, 0, np.random.randint(0,255)), (np.random.randint(1,255), np.random.randint(2,255), np.random.randint(3,128)), (True, False, False))
    random_image = Image.fromarray(np.uint8(array))
    return random_image


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()

# https://stackoverflow.com/q/354038
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def parse_prompt(prompt):
    """Prompts can either just be text, be a text:weight pair, or a text:weight:stop triple"""

    # defaults
    textPrompt = prompt
    weight = 1
    stop = float('-inf')

    # try to parse numbers from the right but stop as soon as that fails
    extra_numbers = []
    keep_going = True

    while len(extra_numbers) < 2 and keep_going:
        vals = textPrompt.rsplit(':', 1)
        if len(vals) > 1 and is_number(vals[1]):
            extra_numbers.append(float(vals[1]))
            textPrompt = vals[0]
        else:
            keep_going = False

    # print(f"parsed nums is {textPrompt}, {extra_numbers}")

    # if there is only 1 number, that becomes the weight
    if len(extra_numbers) == 1:
        weight = extra_numbers[0]
    # if there are two numbers it is weight and stop (stored backwards)
    elif len(extra_numbers) == 2:
        weight = extra_numbers[1]
        stop = extra_numbers[0]

    # print(f"parsed vals is {textPrompt}, {weight}, {stop}")
    return textPrompt, weight, stop

from typing import cast, Dict, List, Optional, Tuple, Union

# override class to get padding_mode
class MyRandomPerspective(K.RandomPerspective):
    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, _, height, width = input.shape
        transform = cast(torch.Tensor, transform)
        return kornia.geometry.transform.warp_perspective(
            input, transform, (height, width), mode=self.flags["resample"].name.lower(),
            align_corners=self.flags["align_corners"], padding_mode=global_padding_mode
        )

global_fill_color=None;
# override class to get fill color
class MyRandomAffine(K.RandomAffine):
    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, _, height, width = input.shape
        transform = cast(torch.Tensor, transform)
        return kornia.geometry.transform.warp_affine(
            input,
            transform[:, :2, :],
            (height, width),
            self.flags["resample"].name.lower(),
            align_corners=self.flags["align_corners"],
            padding_mode="fill",
            fill_value=global_fill_color
        )

class MyRandomPerspectivePadded(K.RandomPerspective):
    def apply_transform(
        self, input: torch.Tensor, params: Dict[str, torch.Tensor], transform: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        _, _, height, width = input.shape
        transform = cast(torch.Tensor, transform)
        return kornia.geometry.transform.warp_perspective(
            input, transform, (height, width), mode=self.flags["resample"].name.lower(),
            align_corners=self.flags["align_corners"],
            padding_mode="fill",
            fill_value=global_fill_color
        )


cached_spot_indexes = {}
def fetch_spot_indexes(sideX, sideY):
    global global_spot_file

    # make sure image is loaded if we need it
    cache_key = (sideX, sideY)

    if cache_key not in cached_spot_indexes:
        if global_spot_file is not None:
            mask_image = Image.open(global_spot_file)
        elif global_aspect_width != 1:
            mask_image = Image.open("inputs/spot_wide.png")
        else:
            mask_image = Image.open("inputs/spot_square.png")
        # this is a one channel mask
        mask_image = mask_image.convert('RGB')
        mask_image = mask_image.resize((sideX, sideY), Image.LANCZOS)
        mask_image_tensor = TF.to_tensor(mask_image)
        # print("ONE CHANNEL ", mask_image_tensor.shape)
        mask_indexes = mask_image_tensor.ge(0.5).to(device)
        # print("GE ", mask_indexes.shape)
        # sys.exit(0)
        mask_indexes_off = mask_image_tensor.lt(0.5).to(device)
        cached_spot_indexes[cache_key] = [mask_indexes, mask_indexes_off]

    return cached_spot_indexes[cache_key]

# n = torch.ones((3,5,5))
# f = generate.fetch_spot_indexes(5, 5)
# f[0].shape = [60,3]

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        global global_aspect_width

        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cutn_zoom = int(0.6*cutn)
        self.cut_pow = cut_pow
        self.transforms = None

        augmentations = []
        # if global_aspect_width != 1:
        #     augmentations.append(K.RandomCrop(size=(self.cut_size,self.cut_size), p=1.0, cropping_mode="resample", return_transform=True))
        augmentations.append(MyRandomPerspective(distortion_scale=0.40, p=0.7, return_transform=True))
        augmentations.append(K.RandomResizedCrop(size=(self.cut_size,self.cut_size), scale=(0.25,0.95),  ratio=(0.85,1.2), cropping_mode='resample', p=1.0, return_transform=True))
        augmentations.append(K.ColorJitter(hue=0.1, saturation=0.1, p=0.8, return_transform=True))
        self.augs_zoom = nn.Sequential(*augmentations)

        augmentations = []
        if global_aspect_width == 1:
            n_s = 0.95
            n_t = (1-n_s)/2
            augmentations.append(MyRandomAffine(degrees=0, translate=(n_t, n_t), scale=(n_s, n_s), p=1.0, return_transform=True))
        elif global_aspect_width > 1:
            n_s = 1/global_aspect_width
            n_t = (1-n_s)/2
            augmentations.append(MyRandomAffine(degrees=0, translate=(0, n_t), scale=(0.9*n_s, n_s), p=1.0, return_transform=True))
        else:
            n_s = global_aspect_width
            n_t = (1-n_s)/2
            augmentations.append(MyRandomAffine(degrees=0, translate=(n_t, 0), scale=(0.9*n_s, n_s), p=1.0, return_transform=True))

        # augmentations.append(K.CenterCrop(size=(self.cut_size,self.cut_size), p=1.0, cropping_mode="resample", return_transform=True))
        augmentations.append(K.CenterCrop(size=self.cut_size, cropping_mode='resample', p=1.0, return_transform=True))
        augmentations.append(MyRandomPerspectivePadded(distortion_scale=0.20, p=0.7, return_transform=True))
        augmentations.append(K.ColorJitter(hue=0.1, saturation=0.1, p=0.8, return_transform=True))
        self.augs_wide = nn.Sequential(*augmentations)

        self.noise_fac = 0.1
        
        # Pooling
        self.av_pool = nn.AdaptiveAvgPool2d((self.cut_size, self.cut_size))
        self.max_pool = nn.AdaptiveMaxPool2d((self.cut_size, self.cut_size))

    def forward(self, input, spot=None):
        global global_aspect_width, cur_iteration
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        mask_indexes = None

        if spot is not None:
            spot_indexes = fetch_spot_indexes(self.cut_size, self.cut_size)
            if spot == 0:
                mask_indexes = spot_indexes[1]
            else:
                mask_indexes = spot_indexes[0]
            # print("Mask indexes ", mask_indexes)

        for _ in range(self.cutn):
            # Pooling
            cutout = (self.av_pool(input) + self.max_pool(input))/2

            if mask_indexes is not None:
                cutout[0][mask_indexes] = 0.0 # 0.5

            if global_aspect_width != 1:
                if global_aspect_width > 1:
                    cutout = kornia.geometry.transform.rescale(cutout, (1, global_aspect_width))
                else:
                    cutout = kornia.geometry.transform.rescale(cutout, (1/global_aspect_width, 1))

            # if cur_iteration % 50 == 0 and _ == 0:
            #     print(cutout.shape)
            #     TF.to_pil_image(cutout[0].cpu()).save(f"cutout_im_{cur_iteration:02d}_{spot}.png")

            cutouts.append(cutout)

        if self.transforms is not None:
            # print("Cached transforms available")
            batch1 = kornia.geometry.transform.warp_perspective(torch.cat(cutouts[:self.cutn_zoom], dim=0), self.transforms[:self.cutn_zoom],
                (self.cut_size, self.cut_size), padding_mode=global_padding_mode)
            batch2 = kornia.geometry.transform.warp_perspective(torch.cat(cutouts[self.cutn_zoom:], dim=0), self.transforms[self.cutn_zoom:],
                (self.cut_size, self.cut_size), padding_mode="fill", fill_value=global_fill_color)
            batch = torch.cat([batch1, batch2])
            # if cur_iteration < 2:
            #     for j in range(4):
            #         TF.to_pil_image(batch[j].cpu()).save(f"cached_im_{cur_iteration:02d}_{j:02d}_{spot}.png")
            #         j_wide = j + self.cutn_zoom
            #         TF.to_pil_image(batch[j_wide].cpu()).save(f"cached_im_{cur_iteration:02d}_{j_wide:02d}_{spot}.png")
        else:
            batch1, transforms1 = self.augs_zoom(torch.cat(cutouts[:self.cutn_zoom], dim=0))
            batch2, transforms2 = self.augs_wide(torch.cat(cutouts[self.cutn_zoom:], dim=0))
            # print(batch1.shape, batch2.shape)
            batch = torch.cat([batch1, batch2])
            # print(batch.shape)
            self.transforms = torch.cat([transforms1, transforms2])
            ## batch, self.transforms = self.augs(torch.cat(cutouts, dim=0))
            # if cur_iteration < 4:
            #     for j in range(4):
            #         TF.to_pil_image(batch[j].cpu()).save(f"live_im_{cur_iteration:02d}_{j:02d}_{spot}.png")
            #         j_wide = j + self.cutn_zoom
            #         TF.to_pil_image(batch[j_wide].cpu()).save(f"live_im_{cur_iteration:02d}_{j_wide:02d}_{spot}.png")

        # print(batch.shape, self.transforms.shape)
        
        if self.noise_fac:
            facs = batch.new_empty([self.cutn, 1, 1, 1]).uniform_(0, self.noise_fac)
            batch = batch + facs * torch.randn_like(batch)
        return batch


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)

def rebuild_optimisers(args):
    global best_loss, best_iter, best_z, num_loss_drop, max_loss_drops, iter_drop_delay
    global drawer, filters

    drop_divisor = 10 ** num_loss_drop
    new_opts = drawer.get_opts(drop_divisor)
    if new_opts == None:
        # legacy

        dropped_learning_rate = args.learning_rate/drop_divisor;
        # print(f"Optimizing with {args.optimiser} set to {dropped_learning_rate}")

        #temporary hack
        if args.init_image and args.drawer=="vdiff":
            dropped_learning_rate = 0.01/drop_divisor

        # Set the optimiser
        to_optimize = [ drawer.get_z() ]
        if args.optimiser == "Adam":
            opt = optim.Adam(to_optimize, lr=dropped_learning_rate)        # LR=0.1
        elif args.optimiser == "AdamW":
            opt = optim.AdamW(to_optimize, lr=dropped_learning_rate)       # LR=0.2
        elif args.optimiser == "Adagrad":
            opt = optim.Adagrad(to_optimize, lr=dropped_learning_rate) # LR=0.5+
        elif args.optimiser == "Adamax":
            opt = optim.Adamax(to_optimize, lr=dropped_learning_rate)  # LR=0.5+?
        elif args.optimiser == "DiffGrad":
            opt = DiffGrad(to_optimize, lr=dropped_learning_rate)      # LR=2+?
        elif args.optimiser == "AdamP":
            opt = AdamP(to_optimize, lr=dropped_learning_rate)     # LR=2+?
        # elif args.optimiser == "RAdam":
        #     opt = RAdam(to_optimize, lr=dropped_learning_rate)     # LR=2+?

        new_opts = [opt]

    return new_opts

# used for target image
def fetch_images(preprocess, image_files):
    images = []

    for filename in image_files:
        image = preprocess(Image.open(filename).convert("RGB"))
        images.append(image)

    return images

def do_image_features(model, images, image_mean, image_std):
    image_input = torch.tensor(np.stack(images)).cuda()
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()

    return image_features

# note: this should probably be split into a setup and a session init
def do_init(args):
    global opts, perceptors, normalize, cutoutsTable, cutoutSizeTable
    global z_orig, im_targets, z_labels, init_image_tensor, target_image_tensor
    global gside_X, gside_Y, overlay_image_rgba, overlay_image_rgba_list, init_image_rgba_list
    global pmsTable, pmsImageTable, pmsTargetTable, pImages, device, spotPmsTable, spotOffPmsTable
    global drawer, filters
    global lossGlobals, global_cached_png_info, global_seed_used

    reset_session_globals()

    # do seed first!
    if args.seed is None:
        seed = torch.seed()
    elif isinstance(args.seed, int):
        seed = args.seed
    elif isinstance(args.seed, str) and args.seed.isdigit():
        seed = int(args.seed)
    else:
        # deterministic 32 bit int from string
        # https://stackoverflow.com/a/44556106/1010653
        e_str = args.seed.encode()
        hash_digest = hashlib.sha512(e_str).digest()
        seed = int.from_bytes(hash_digest, 'big') % 0x100000000
    int_seed = int(seed)%(2**30)
    print('Using seed:', seed)
    global_seed_used = seed
    torch.manual_seed(seed)
    np.random.seed(int_seed)
    random.seed(int_seed)

    # set device only once
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    drawer = class_table[args.drawer](args)
    drawer.load_model(args, device)
    num_resolutions = drawer.get_num_resolutions()

    # print("-----------> NUMR ", num_resolutions)
    #as of torch 1.8, jit produces errors. The below code no longer works with 1.10
    #jit = True if float(torch.__version__[:3]) < 1.8 else False
    jit = False

    if num_resolutions!=None:
        f = 2**(num_resolutions - 1)
        toksX, toksY = args.size[0] // f, args.size[1] // f
        sideX, sideY = toksX * f, toksY * f
    else:
        sideX, sideY = args.size[0], args.size[1]

    # save sideX, sideY in globals (need if using overlay)
    gside_X = sideX
    gside_Y = sideY

    # model loading optimization: if all models are loaded keep things as they are
    if set(args.clip_models) <= set(perceptors.keys()):
        print("All CLIP models already loaded: ", args.clip_models)
    else:
        # TODO: unload models?
        perceptors = {}
        for clip_model in args.clip_models:
            perceptor = get_clip_perceptor(clip_model, device)
            perceptors[clip_model] = perceptor

    # now separately setup cuts
    for clip_model in args.clip_models:
        perceptor = perceptors[clip_model]
        cut_size = perceptor.input_resolution
        cutoutSizeTable[clip_model] = cut_size
        if not cut_size in cutoutsTable:    
            make_cutouts = MakeCutouts(cut_size, args.num_cuts, cut_pow=args.cut_pow)
            cutoutsTable[cut_size] = make_cutouts

    filters = None
    if args.filters is not None:
        filter_names = args.filters.split(",")
        filter_names = [f.strip() for f in filter_names]
        filterClasses = []
        for filt in filter_names:
            filt_name, weight, stop = parse_prompt(filt)
            if filt_name not in filters_class_table:
                raise ValueError(f"Requested filter not found, aborting: {filt_name}")
            filtClass = filters_class_table[filt_name]
            # do special initializations here
            try:
                filtInstance = filtClass(args, device=device)
                filterClasses.append({"filter":filtInstance, "weight": weight})
            except TypeError as e:
                print(f'error in initializing {filtClass} - this message is to provide information')
                raise TypeError(e)
        filters = filterClasses

    init_image_tensor = None
    target_image_tensor = None

    # Image initialisation
    if args.init_image or args.init_noise:
        # setup init image wih pil
        # first - always start with noise or blank
        if args.init_noise == 'pixels':
            img = random_noise_image(args.size[0], args.size[1])
        elif args.init_noise == 'gradient':
            img = random_gradient_image(args.size[0], args.size[1])
        elif args.init_noise == 'snow':
            img = old_random_noise_image(args.size[0], args.size[1])
        else:
            img = Image.new(mode="RGB", size=(args.size[0], args.size[1]), color=(255, 255, 255))
        starting_image = img.convert('RGB')
        starting_image = starting_image.resize((sideX, sideY), Image.LANCZOS)

        if args.init_image:
            # now we might overlay an init image
            filelist = None
            if 'http' in args.init_image:
                init_images = [Image.open(urlopen(args.init_image))]
            else:
                filelist = real_glob(args.init_image)
                init_images = [Image.open(f) for f in filelist]

            init_image_rgba_list = []
            for init_image in init_images:
                # this version is needed potentially for the loss function
                init_image_rgb = init_image.convert('RGB')
                init_image_rgb = init_image_rgb.resize((sideX, sideY), Image.LANCZOS)
                init_image_tensor = TF.to_tensor(init_image_rgb)
                init_image_tensor = init_image_tensor.to(device).unsqueeze(0)

                # this version gets overlaid on the background (noise)
                init_image_rgba = init_image.convert('RGBA')
                init_image_rgba = init_image_rgba.resize((sideX, sideY), Image.LANCZOS)
                top_image = init_image_rgba.copy()
                if args.init_image_alpha and args.init_image_alpha >= 0:
                    top_image.putalpha(args.init_image_alpha)
                cur_start_image = starting_image.copy()
                cur_start_image.paste(top_image, (0, 0), top_image)
                init_image_rgba_list.append(cur_start_image)

            starting_image = init_image_rgba_list[0]

            save_image(init_image_tensor,"init_image_tensor.png")
            drawer.init_from_tensor(init_image_tensor * 2 - 1)
            z_orig = drawer.get_z_copy()
        else:
            starting_image.save("starting_image.png")
            starting_tensor = TF.to_tensor(starting_image)
            init_tensor = starting_tensor.to(device).unsqueeze(0)
            drawer.init_from_tensor(init_tensor * 2 - 1)

    else:
        drawer.init_from_tensor(init_tensor=None)
        # this is the old vqgan version [need to patch vqgan to do this?]
        # drawer.rand_init(toksX, toksY)

    if args.overlay_image is not None:
        # todo: maybe split this up on pipes and whatnot
        overlay_image_rgba_list = []
        if 'http' in args.overlay_image:
            overlay_images = [Image.open(urlopen(args.overlay_image))]
        else:
            filelist = real_glob(args.overlay_image)
            overlay_images = [Image.open(f) for f in filelist]

        for overlay_image in overlay_images:
            overlay_image_rgba = overlay_image.convert('RGBA')
            overlay_image_rgba = overlay_image_rgba.resize((sideX, sideY), Image.LANCZOS)
            if args.overlay_alpha:
                overlay_image_rgba.putalpha(args.overlay_alpha)
            overlay_image_rgba_list.append(overlay_image_rgba)

        overlay_image_rgba_list[0].save('overlay_image0.png')

    global_cached_png_info = None

    pmsTable = {}
    pmsImageTable = {}
    pmsTargetTable = {}
    spotPmsTable = {}
    spotOffPmsTable = {}
    for clip_model in args.clip_models:
        pmsTable[clip_model] = []
        pmsImageTable[clip_model] = []
        pmsTargetTable[clip_model] = []
        spotPmsTable[clip_model] = []
        spotOffPmsTable[clip_model] = []

    drawer_clip_target = None
    if hasattr(drawer, 'clip_model') and drawer.clip_model is not None:
        print(f"drawer {drawer} needs {drawer.clip_model}")
        drawer_clip_target = drawer.clip_model
    # NR: Weights / blending
    allpromptembeds = []
    allweights = []

    if args.target_images is not None:
        if args.animation_dir is not None:
            for clip_model in args.clip_models:
                pmsTarget = pmsTargetTable[clip_model]
                perceptor = perceptors[clip_model]

                input_resolution = perceptor.input_resolution
                # print(f"Running {clip_model} at {input_resolution}")
                preprocess = Compose([
                    Resize(input_resolution, interpolation=Image.BICUBIC),
                    CenterCrop(input_resolution),
                    ToTensor()
                ])
                image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
                image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

                input_files = []
                for target_image in args.target_images:
                    f1, weight, stop = parse_prompt(target_image)
                    infiles = real_glob(f1)
                    input_files.extend(infiles)

                for path in input_files:
                    images = fetch_images(preprocess, [path])
                    features = do_image_features(perceptor, images, image_mean, image_std)
                    pmsTarget.append(Prompt(features, weight, stop).to(device))
        else:
            for clip_model in args.clip_models:
                pMs = pmsTable[clip_model]
                perceptor = perceptors[clip_model]

                input_resolution = perceptor.input_resolution
                # print(f"Running {clip_model} at {input_resolution}")
                preprocess = Compose([
                    Resize(input_resolution, interpolation=Image.BICUBIC),
                    CenterCrop(input_resolution),
                    ToTensor()
                ])
                image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
                image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

                input_files = []
                for target_image in args.target_images:
                    f1, weight, stop = parse_prompt(target_image)
                    # print("Target parse ", target_image, "to", f1)
                    if 'http' in f1:
                        # note: this is currently untested...
                        infile = urlopen(f1)
                        input_files.append(infile)
                    else:
                        infiles = real_glob(f1)
                        input_files.extend(infiles)

                print(input_files)
                images = fetch_images(preprocess, input_files);

                features = do_image_features(perceptor, images, image_mean, image_std)
                if clip_model == drawer_clip_target:
                    allpromptembeds.append(features)
                    allweights.append(weight)
                pMs.append(Prompt(features, weight, stop).to(device))

    if args.image_labels is not None:
        z_labels = []
        filelist = real_glob(args.image_labels)
        cur_labels = []
        for image_label in filelist:
            image_label = Image.open(image_label)
            image_label_rgb = image_label.convert('RGB')
            image_label_rgb = image_label_rgb.resize((sideX, sideY), Image.LANCZOS)
            image_label_rgb_tensor = TF.to_tensor(image_label_rgb)
            image_label_rgb_tensor = image_label_rgb_tensor.to(device).unsqueeze(0) * 2 - 1
            z_label = drawer.get_z_from_tensor(image_label_rgb_tensor)
            cur_labels.append(z_label)
        image_embeddings = torch.stack(cur_labels)
        print("Processing labels: ", image_embeddings.shape)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        image_embeddings = image_embeddings.mean(dim=0)
        image_embeddings /= image_embeddings.norm()
        z_labels.append(image_embeddings.unsqueeze(0))

    if z_orig is not None:
        z_orig = drawer.get_z_copy()

    # normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
    #                                   std=[0.26862954, 0.26130258, 0.27577711])

    # CLIP tokenize/encode
    for prompt in args.prompts:
        for clip_model in args.clip_models:
            pMs = pmsTable[clip_model]
            perceptor = perceptors[clip_model]
            txt, weight, stop = parse_prompt(prompt)
            if txt[0] == '=':
                # hack for now to test pseudo encode shim
                txt = txt[1:]
                print(f"--> {clip_model} encoding {txt} with stops")
                actual_tokens = clip.tokenize(txt).to(device)
                stops = actual_tokens.argmax(dim=-1) - 1
                embed = perceptor.encode_text(actual_tokens, stops).float()
            else:
                # print(f"--> {clip_model} normal encoding {txt}")
                embed = perceptor.encode_text(txt).float()
            if clip_model == drawer_clip_target:
                allpromptembeds.append(embed)
                allweights.append(weight)
            pMs.append(Prompt(embed, weight, stop).to(device))

    if drawer_clip_target is not None:
        if args.drawer=="vdiff" and args.vdiff_model[:7] == "cc12m_1":
            target_embeds = torch.cat(allpromptembeds)
            allweights = torch.tensor(allweights, dtype=torch.float, device=device)
            clip_embed = F.normalize(target_embeds.mul(allweights[:, None]).sum(0, keepdim=True), dim=-1)
            print(f"clip_embed for drawer {drawer} is {clip_embed.shape}")
            drawer.sample_state[3] = {"clip_embed":clip_embed}

    for vect_prompt in args.vector_prompts:
        f1, weight, stop = parse_prompt(vect_prompt)
        # vect_promts are by nature tuned to 10% of a normal prompt
        weight = 0.1 * weight
        if 'http' in f1:
            # note: this is currently untested...
            infile = None
            infile_handle = urlopen(f1)
        elif 'json' in f1:
            infile = f1
        else:
            infile = f"vectors/{f1}.json"
            if not os.path.exists(infile):
                infile = f"pixray/vectors/{f1}.json"
        if infile:
            with open(infile) as f_in:
                vect_table = json.load(f_in)
        else:
            vect_table = json.load(infile_handle)
        for clip_model in args.clip_models:
            if clip_model not in vect_table:
                print(f"WARNING: no vector for {clip_model} in {f1}!")
                print("Continuing without this vector... (BUT THIS RESULT MIGHT NOT BE WHAT YOU WANT 😬)")
                # time.sleep(3)
                continue
            pMs = pmsTable[clip_model]
            v = np.array(vect_table[clip_model])
            embed = torch.FloatTensor(v).to(device).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.spot_prompts:
        for clip_model in args.clip_models:
            pMs = spotPmsTable[clip_model]
            perceptor = perceptors[clip_model]
            txt, weight, stop = parse_prompt(prompt)
            embed = perceptor.encode_text(txt).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

    for prompt in args.spot_prompts_off:
        for clip_model in args.clip_models:
            pMs = spotOffPmsTable[clip_model]
            perceptor = perceptors[clip_model]
            txt, weight, stop = parse_prompt(prompt)
            embed = perceptor.encode_text(txt).float()
            pMs.append(Prompt(embed, weight, stop).to(device))

    for label in args.labels:
        for clip_model in args.clip_models:
            pMs = pmsTable[clip_model]
            perceptor = perceptors[clip_model]
            txt, weight, stop = parse_prompt(label)
            texts = [template.format(txt) for template in imagenet_templates] #format with class
            # print(f"Tokenizing all of {texts}")
            # texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = perceptor.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            pMs.append(Prompt(class_embedding.unsqueeze(0), weight, stop).to(device))

    for clip_model in args.clip_models:
        pImages = pmsImageTable[clip_model]
        for path in args.image_prompts:
            img = Image.open(path)
            pil_image = img.convert('RGB')
            img = resize_image(pil_image, (sideX, sideY))
            pImages.append(TF.to_tensor(img).unsqueeze(0).to(device))

    for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
        gen = torch.Generator().manual_seed(seed)
        embed = torch.empty([1, perceptor.output_dim]).normal_(generator=gen)
        pMs.append(Prompt(embed, weight).to(device))

    #custom loss 
    if args.custom_loss is not None:
        custom_losses = args.custom_loss.split(",")
        custom_losses = [loss.strip() for loss in custom_losses]
        custom_loss_names = args.custom_loss
        lossClasses = []
        for loss_chunk in custom_losses:
            # check for special delimiter
            if loss_chunk.find('->') > 0:
                parts = loss_chunk.split('->')
                loss = parts[0]
                instance_args = parts[1:]
            else:
                loss = loss_chunk
                instance_args = []
            loss_name, weight, stop = parse_prompt(loss)
            lossClass = loss_class_table[loss_name]
            # do special initializations here
            try:
                lossInstance = lossClass(device=device)
                lossInstance.instance_settings(instance_args)
                lossClasses.append({"loss":lossInstance, "weight": weight})
            except TypeError as e:
                print(f'error in initializing {lossClass} - this message is to provide information')
                raise TypeError(e)
        args.custom_loss = lossClasses

    #Loss args parse
    if args.custom_loss:
        for t in args.custom_loss:
            args = t["loss"].parse_settings(args)

    #adding globals for loss
    if args.custom_loss is not None and len(args.custom_loss)>0:
        for t in args.custom_loss:
            lossGlobals.update(t["loss"].add_globals(args))
    
    
    opts = rebuild_optimisers(args)

    # Output for the user
    print('Using device:', device)
    print('Optimising using:', args.optimiser)

    if args.prompts:
        print('Using text prompts:', args.prompts)
    if args.spot_prompts:
        print('Using spot prompts:', args.spot_prompts)
    if args.spot_prompts_off:
        print('Using spot off prompts:', args.spot_prompts_off)
    if args.image_prompts:
        print('Using #image prompts:', len(args.image_prompts))
    if args.init_image:
        print(f'Using initial image {args.init_image} ({len(init_image_rgba_list)})')
    if args.noise_prompt_weights:
        print('Noise prompt weights:', args.noise_prompt_weights)
    if args.custom_loss:
        print(f'using custom losses: {str(custom_loss_names)}')

    cur_iteration = 0


# dreaded unsorted globals (for now)
z_orig = None
im_targets = None
z_labels = None
opts = None
drawer = None
filters = None
# normalize = None
init_image_tensor = None
target_image_tensor = None
pmsTable = None
spotPmsTable = None 
spotOffPmsTable = None 
pmsImageTable = None
pmsTargetTable = None
gside_X=None
gside_Y=None
init_image_rgba_list=[]
overlay_image_rgba_list=None
overlay_image_rgba=None
cur_iteration=None
cur_anim_index=None
anim_output_files=[]
anim_cur_zs=[]
anim_next_zs=[]
best_loss = None 
best_iter = None
best_z = None
num_loss_drop = 0
max_loss_drops = 2
iter_drop_delay = 20

# session globals
cutoutsTable = {}
cutoutSizeTable = {}

# persistent globals
perceptors = {}
device=None

#loss globals
lossGlobals = {}

# on re-runs this should reset most important globals
def reset_session_globals():
    global cutoutsTable, cutoutSizeTable
    cutoutsTable = {}
    cutoutSizeTable = {}

def make_gif(args, iter):
    gif_output = os.path.join(args.animation_dir, "anim.gif")
    if os.path.exists(gif_output):
        os.remove(gif_output)
    cmd = ['ffmpeg', '-framerate', '10', '-pattern_type', 'glob',
           '-i', f"{args.animation_dir}/*.png", '-loop', '0', gif_output]
    try:
        output = subprocess.check_output(cmd)
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)

    return gif_output

# !ffmpeg \
#   -framerate 10 -pattern_type glob \
#   -i '{animation_output}/*_*.png' \
#   -loop 0 {animation_output}/final.gif

@torch.no_grad()
def checkdrop(args, iter, losses):
    global best_loss, best_iter, best_z, num_loss_drop, max_loss_drops, iter_drop_delay
    global drawer

    drop_loss_time = False

    loss_sum = sum(losses)
    is_new_best = False
    num_cycles_not_best = 0
    if (loss_sum < best_loss):
        is_new_best = True
        best_loss = loss_sum
        best_iter = iter
        best_z = drawer.get_z_copy()
    else:
        num_cycles_not_best = iter - best_iter
        if num_cycles_not_best >= iter_drop_delay:
            drop_loss_time = True
    return drop_loss_time

# for a release just bake in the version to prevent git subprocess lookup
git_official_release_version = None
git_fallback_version = "v1.7.2+"

# https://stackoverflow.com/a/40170206/1010653
# Return the git revision as a string
def git_version():
    global git_official_release_version, git_fallback_version
    if git_official_release_version is not None:
        return git_official_release_version

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout = subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'describe', '--always'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = git_fallback_version

    return GIT_REVISION

global_cached_png_info=None
def getPngInfo():
    global global_cached_png_info
    if global_cached_png_info is None:
        git_v = git_version()
        info = PngImagePlugin.PngInfo()
        info.add_text("Software", f"pixray ({git_v})")
        # print(global_given_args)
        for k in global_given_args:
            info.add_text(f"pixray_{k}", str(global_given_args[k]))
        info.add_text("pixray_seed_used", str(global_seed_used))
        global_cached_png_info = info
    return global_cached_png_info 

@torch.no_grad()
def checkin(args, iter, losses):
    global drawer, filters
    global best_loss, best_iter, best_z, num_loss_drop, max_loss_drops, iter_drop_delay

    num_cycles_not_best = iter - best_iter

    if losses is not None:
        losses_str = ', '.join(f'{loss.item():2.3g}' for loss in losses)
        writestr = f'iter: {iter}, loss: {sum(losses).item():1.3g}, losses: {losses_str}'
    else:
        writestr = f'iter: {iter}, finished'

    if args.animation_dir is not None:
        writestr = f'anim: {cur_anim_index}/{len(anim_output_files)} {writestr}'
    else:
        writestr = f'{writestr} (-{num_cycles_not_best}=>{best_loss:2.4g})'
    timg = drawer.synth(cur_iteration)
    if filters is not None and len(filters)>0:
        for f in filters:
            filtclass = f["filter"]
            timg, closs = filtclass(timg);

    img = TF.to_pil_image(timg[0].cpu())
    # img = drawer.to_image()
    if cur_anim_index is None:
        outfile = get_file_path(args.output_dir, args.output, '.png')
    else:
        outfile = anim_output_files[cur_anim_index]
    img.save(outfile, pnginfo=getPngInfo())
    if cur_anim_index == len(anim_output_files) - 1:
        # save gif
        gif_output = make_gif(args, iter)
        if IS_NOTEBOOK and iter % args.display_every == 0:
            clear_output()
            display.display(display.Image(open(gif_output,'rb').read()))
    if IS_NOTEBOOK and iter % args.display_every == 0:
        if cur_anim_index is None or iter == 0:
            if args.display_clear:
                clear_output()
            display.display(display.Image(outfile))
    tqdm.write(writestr)

def ascend_txt(args):
    global cur_iteration, cur_anim_index, perceptors, cutoutsTable, cutoutSizeTable # normalize, 
    global z_orig, im_targets, z_labels, init_image_tensor, target_image_tensor, drawer
    global pmsTable, pmsImageTable, spotPmsTable, spotOffPmsTable, global_padding_mode, global_fill_color
    global pmsTargetTable
    global lossGlobals

    if args.transparency:
        out, alpha = drawer.synth(cur_iteration, return_transparency=True)
    else:
        out = drawer.synth(cur_iteration)

    result = []

    if filters is not None and len(filters)>0:
        for f in filters:
            filtclass = f["filter"]
            filtweight = f["weight"]
            out, new_losses = filtclass(out);
            if type(new_losses) is not list and type(new_losses) is not tuple:
                result.append(filtweight * new_losses)
            else:
                # warning: this path might be untested by current losses?
                weighted_losses = [(filtweight * l) for l in new_losses]
                result += weighted_losses

    if (cur_iteration%2 == 0):
        global_padding_mode = 'reflection'
    else:
        global_padding_mode = 'border'

    color_fill = random.random()
    # print("Color fill is ", color_fill)
    # color_fill = 1.0
    global_fill_color =torch.tensor([color_fill,color_fill,color_fill], device=device, dtype=torch.float)

    cur_cutouts = {}
    cur_spot_cutouts = {}
    cur_spot_off_cutouts = {}
    for cutoutSize in cutoutsTable:
        make_cutouts = cutoutsTable[cutoutSize]
        cur_cutouts[cutoutSize] = make_cutouts(out)

    if args.spot_prompts:
        for cutoutSize in cutoutsTable:
            cur_spot_cutouts[cutoutSize] = make_cutouts(out, spot=1)

    if args.spot_prompts_off:
        for cutoutSize in cutoutsTable:
            cur_spot_off_cutouts[cutoutSize] = make_cutouts(out, spot=0)

    for clip_model in args.clip_models:
        perceptor = perceptors[clip_model]
        cutoutSize = cutoutSizeTable[clip_model]
        transient_pMs = []

        if args.spot_prompts:
            iii_s = perceptor.encode_image(cur_spot_cutouts[cutoutSize]).float()
            spotPms = spotPmsTable[clip_model]
            for prompt in spotPms:
                result.append(prompt(iii_s))

        if args.spot_prompts_off:
            iii_so = perceptor.encode_image(cur_spot_off_cutouts[cutoutSize]).float()
            spotOffPms = spotOffPmsTable[clip_model]
            for prompt in spotOffPms:
                result.append(prompt(iii_so))

        iii = perceptor.encode_image(cur_cutouts[cutoutSize]).float()

        pMs = pmsTable[clip_model]
        for prompt in pMs:
            result.append(prompt(iii))

        # add target frame prompts if applicable
        if cur_anim_index is not None and len(pmsTargetTable[clip_model]) > 0:
            num_anim_frames = len(pmsTargetTable[clip_model])
            pmsTarget = [ pmsTargetTable[clip_model][cur_anim_index % num_anim_frames] ]
            for prompt in pmsTarget:
                result.append(prompt(iii))

        # If there are image prompts we make cutouts for those each time
        # so that they line up with the current cutouts from augmentation
        make_cutouts = cutoutsTable[cutoutSize]

        # if animating select one pImage, otherwise use them all
        if cur_anim_index is not None and len(pmsImageTable[clip_model]) > 0:
            num_anim_frames = len(pmsImageTable[clip_model])
            pImages = [ pmsImageTable[clip_model][cur_anim_index % num_anim_frames] ]
        else:
            pImages = pmsImageTable[clip_model]
        
        for timg in pImages:
            # note: this caches and reuses the transforms - a bit of a hack but it works

            if args.image_prompt_shuffle:
                # print("Disabling cached transforms")
                make_cutouts.transforms = None

            # print("Building throwaway image prompts")
            # new way builds throwaway Prompts
            batch = make_cutouts(timg)
            embed = perceptor.encode_image(batch).float()
            if args.image_prompt_weight is not None:
                transient_pMs.append(Prompt(embed, args.image_prompt_weight).to(device))
            else:
                transient_pMs.append(Prompt(embed).to(device))

        for prompt in transient_pMs:
            result.append(prompt(iii))


    for cutoutSize in cutoutsTable:
        # clear the transform "cache"
        make_cutouts = cutoutsTable[cutoutSize]
        make_cutouts.transforms = None

    if args.image_labels is not None:
        for z_label in z_labels:
            f = drawer.get_z().reshape(1,-1)
            f2 = z_label.reshape(1,-1)
            cur_loss = spherical_dist_loss(f, f2) * args.image_label_weight
            result.append(cur_loss)

    # main init_weight uses spherical loss
    if args.init_weight:
        f = drawer.get_z().reshape(1,-1)
        f2 = z_orig.reshape(1,-1)
        cur_loss = spherical_dist_loss(f, f2) * args.init_weight
        result.append(cur_loss[0])

    # these three init_weight variants offer mse_loss, mse_loss in pixel space, and cos loss
    if args.init_weight_dist:
        cur_loss = F.mse_loss(drawer.get_z(), z_orig) * args.init_weight_dist / 2
        result.append(cur_loss)

    if args.init_weight_pix:
        if init_image_tensor is None:
            print("OOPS IIT is 0")
        else:
            cur_loss = F.l1_loss(out, init_image_tensor) * args.init_weight_pix / 2
            result.append(cur_loss)

    if args.init_weight_cos:
        f = drawer.get_z().reshape(1,-1)
        f2 = z_orig.reshape(1,-1)
        y = torch.ones_like(f[0])
        cur_loss = F.cosine_embedding_loss(f, f2, y) * args.init_weight_cos
        result.append(cur_loss)
    
    needed_globals = {
        # used to be for palette loss - now left as an example
        "cur_iteration":cur_iteration,
        "embeds": iii,
    }

    if args.transparency:
        result.append(args.alpha_weight*torch.mean(alpha))
    
    if args.custom_loss is not None and len(args.custom_loss)>0:
        for t in args.custom_loss:
            lossclass = t["loss"]
            lossweight = t["weight"]
            new_losses = lossclass.get_loss(cur_cutouts, out, args, globals = needed_globals, lossGlobals = lossGlobals)
            if type(new_losses) is not list and type(new_losses) is not tuple:
                result.append(lossweight * new_losses)
            else:
                # warning: this path might be untested by current losses?
                weighted_losses = [(lossweight * l) for l in new_losses]
                result += weighted_losses

    if args.make_video:    
        img = np.array(out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8))[:,:,:]
        img = np.transpose(img, (1, 2, 0))
        imageio.imwrite(f'./steps/frame_{cur_iteration:04d}.png', np.array(img))

    return result

def re_average_z(args):
    global gside_X, gside_Y
    global device, drawer

    # old_z = z.clone()
    cur_z_image = drawer.to_image()
    cur_z_image = cur_z_image.convert('RGB')
    if overlay_image_rgba:
        # print("applying overlay image")
        cur_z_image.paste(overlay_image_rgba, (0, 0), mask=overlay_image_rgba)
        # cur_z_image.save("overlaid.png")
    cur_z_image = cur_z_image.resize((gside_X, gside_Y), Image.LANCZOS)
    drawer.reapply_from_tensor(TF.to_tensor(cur_z_image).to(device).unsqueeze(0) * 2 - 1)

def init_anim_z(args, init_rgba):
    global gside_X, gside_Y
    global device, drawer

    cur_z_image = init_rgba.copy()
    drawer.reapply_from_tensor(TF.to_tensor(cur_z_image).to(device).unsqueeze(0) * 2 - 1)

# torch.autograd.set_detect_anomaly(True)

def apply_overlay(args, cur_it):
    return args.overlay_image is not None and \
            (cur_it % args.overlay_every) == args.overlay_offset and \
            ((args.overlay_until is None) or (cur_it < args.overlay_until))

def train(args, cur_it):
    global drawer, opts
    global best_loss, best_iter, best_z, num_loss_drop, max_loss_drops, iter_drop_delay
    global overlay_image_rgba, overlay_image_rgba_list, cur_anim_index, init_image_rgba_list
    
    rebuild_opts_when_done = False

    lossAll = None
    if cur_it < args.iterations:
        # this is awkward, but train is in also in charge of saving, so...
        rebuild_opts_when_done = False

        for opt in opts:
            # opt.zero_grad(set_to_none=True)
            opt.zero_grad()

        if cur_it == 0 and len(init_image_rgba_list) > 0:
            if cur_anim_index is not None:
                num_anim_frames = len(init_image_rgba_list)
                init_anim_z(args, init_image_rgba_list[cur_anim_index % num_anim_frames])

        if apply_overlay(args, cur_it):
            if cur_anim_index is not None:
                num_anim_frames = len(overlay_image_rgba_list)
                overlay_image_rgba = overlay_image_rgba_list[cur_anim_index % num_anim_frames]
            re_average_z(args)

        # num_batches = args.batches * (num_loss_drop + 1)
        num_batches = args.batches
        for i in range(num_batches):
            lossAll = ascend_txt(args)

            if i == 0:
                if cur_anim_index is None or cur_anim_index == 0:
                    if cur_it in args.learning_rate_drops:
                        print("Dropping learning rate")
                        rebuild_opts_when_done = True
                    else:
                        did_drop = checkdrop(args, cur_it, lossAll)
                        if args.auto_stop is True:
                            rebuild_opts_when_done = did_drop

            if i == 0 and cur_it % args.save_every == 0:
                checkin(args, cur_it, lossAll)

            loss = sum(lossAll)
            loss.backward()

        for opt in opts:
            opt.step()

        drawer.clip_z()

    if args.drawer == "vdiff" and cur_it>=1:
        lr = drawer.sample_state[6][cur_it] / drawer.sample_state[5][cur_it]
        drawer.x = drawer.makenoise(cur_it)
        drawer.x.requires_grad_()
        to_optimize = [ drawer.x ]
        opt = optim.Adam(to_optimize, lr=min(lr*0.001,0.01))
        opts = [opt]
    if cur_it == args.iterations:
        # this resetting to best is currently disabled
        # drawer.set_z(best_z)
        checkin(args, cur_it, lossAll)
        return False
    if rebuild_opts_when_done:
        num_loss_drop = num_loss_drop + 1
        # this resetting to best is currently disabled
        # drawer.set_z(best_z)
        # always checkin (and save) after resetting z
        # checkin(args, cur_it, lossAll)
        if num_loss_drop > max_loss_drops:
            return False
        best_iter = cur_it
        best_loss = 1e20
        opts = rebuild_optimisers(args)
    return True

imagenet_templates = [
    "itap of a {}.",
    "a bad photo of the {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
]

def check_new_filelist(filelist_old_source, filelist_old, filelist_cur_source, filelist_cur):
    if filelist_old_source is None:
        print(f"==> setting animation filelist to {filelist_cur_source} ({len(filelist_cur)} files)")
        return filelist_cur_source, filelist_cur
    elif len(filelist_old) > len(filelist_cur):
        print(f"==> anim filelist {filelist_cur_source} only has {len(filelist_cur)} files - sticking with {filelist_old_source}")
        return filelist_old_source, filelist_old
    elif len(filelist_old) == len(filelist_cur):
        print(f"==> anim filelist {filelist_cur_source} also has {len(filelist_cur)} files - sticking with {filelist_old_source}")
        return filelist_old_source, filelist_old
    elif len(filelist_old) < len(filelist_cur):
        print(f"==> anim filelist {filelist_cur_source} has {len(filelist_cur)} files - switching from {filelist_old_source}")
        return filelist_cur_source, filelist_cur

# return only once to run only one iteration
# returns True when complete, False otherwise
def do_run(args, return_display=False):
    global cur_iteration, cur_anim_index
    global anim_cur_zs, anim_next_zs, anim_output_files

    if args.animation_dir is not None:
        # we already have z_targets. setup some sort of global ring
        # we need something like
        # copies of all the current z's (they can all start off all as copies)
        # a list of all the output filenames
        #
        if not os.path.exists(args.animation_dir):
            os.mkdir(args.animation_dir)
        filelist = []
        filelist_source = None
        if args.overlay_image is not None:
            filelist_cur = real_glob(args.overlay_image)
            filelist_source, filelist = check_new_filelist(filelist_source, filelist, "overlay_images", filelist_cur)
        if args.target_images is not None and len(args.target_images) > 0:
            filelist_cur = []
            for target_image in args.target_images:
                f1, weight, stop = parse_prompt(target_image)
                infiles = real_glob(f1)
                filelist_cur.extend(infiles)
            filelist_source, filelist = check_new_filelist(filelist_source, filelist, "target_images", filelist_cur)
        if args.init_image is not None:
            filelist_cur = real_glob(args.init_image)
            filelist_source, filelist = check_new_filelist(filelist_source, filelist, "init_images", filelist_cur)
        if args.image_prompts is not None and len(args.image_prompts) > 0:
            filelist = args.image_prompts
            filelist_source, filelist = check_new_filelist(filelist_source, filelist, "image_prompts", filelist_cur)
        num_anim_frames = len(filelist)
        for target_image in filelist:
            basename = os.path.basename(target_image)
            target_output = os.path.join(args.animation_dir, basename)
            anim_output_files.append(target_output)
        for i in range(num_anim_frames):
            cur_z = drawer.get_z_copy()
            anim_cur_zs.append(cur_z)
            anim_next_zs.append(None)

        step_iteration = 0

        with tqdm() as pbar:
            while True:
                cur_images = []
                for i in range(num_anim_frames):
                    # do merge frames here from cur->next when we are ready to be fancy
                    cur_anim_index = i
                    # anim_cur_zs[cur_anim_index] = anim_next_zs[cur_anim_index]
                    cur_iteration = step_iteration
                    drawer.set_z(anim_cur_zs[cur_anim_index])
                    for j in range(args.save_every):
                        keep_going = train(args, cur_iteration)
                        cur_iteration += 1
                        pbar.update()
                    # anim_next_zs[cur_anim_index] = drawer.get_z_copy()
                    cur_images.append(drawer.to_image())
                step_iteration = step_iteration + args.save_every
                if step_iteration >= args.iterations:
                    break
                # compute the next round of cur_zs here from all the next_zs
                for i in range(num_anim_frames):
                    prev_i = (i + num_anim_frames - 1) % num_anim_frames
                    base_image = cur_images[i].copy()
                    prev_image = cur_images[prev_i].copy().convert('RGBA')
                    prev_image.putalpha(args.animation_alpha)
                    base_image.paste(prev_image, (0, 0), prev_image)
                    # base_image.save(f"overlaid_{i:02d}.png")
                    drawer.reapply_from_tensor(TF.to_tensor(base_image).to(device).unsqueeze(0) * 2 - 1)
                    anim_cur_zs[i] = drawer.get_z_copy()
    else:
        try:
            keep_going = True
            with tqdm() as pbar:
                while keep_going:
                    try:
                        keep_going = train(args, cur_iteration)
                        if cur_iteration == args.iterations:
                            break
                        cur_iteration += 1
                        if not hasattr(args, 'skip_args'):
                            pbar.update()
                        if keep_going and return_display and cur_iteration % args.display_every == 0:
                            # print("Returning after iteration ", cur_iteration)
                            return False
                    except RuntimeError as e:
                        print("Oops: runtime error: ", e)
                        print("Try reducing --num-cuts to save memory")
                        raise e
        except KeyboardInterrupt:
            pass

    if args.make_video:
        do_video(args)

    return True

def do_video(args):
    global cur_iteration

    # Video generation
    init_frame = 1 # This is the frame where the video will start
    last_frame = cur_iteration # You can change to the number of the last frame you want to generate. It will raise an error if that number of frames does not exist.

    min_fps = 10
    max_fps = 60

    total_frames = last_frame-init_frame

    length = 15 # Desired time of the video in seconds

    frames = []
    tqdm.write('Generating video...')
    for i in range(init_frame,last_frame): #
        frames.append(Image.open(f'./steps/frame_{i:04d}.png'))

    #fps = last_frame/10
    fps = np.clip(total_frames/length,min_fps,max_fps)

    from subprocess import Popen, PIPE
    output_file = get_file_path(args.output_dir, args.output, '.mp4')
    p = Popen(['ffmpeg',
               '-y',
               '-f', 'image2pipe',
               '-vcodec', 'png',
               '-r', str(fps),
               '-i',
               '-',
               '-vcodec', 'libx264',
               '-r', str(fps),
               '-pix_fmt', 'yuv420p',
               '-crf', '17',
               '-preset', 'veryslow',
               '-metadata', f'comment={args.prompts}',
               output_file], stdin=PIPE)
    for im in tqdm(frames):
        im.save(p.stdin, 'PNG')
    p.stdin.close()
    p.wait()

# this dictionary is used for settings in the notebook
global_pixray_settings = {}
# this dictionary documents all non-default args
global_given_args = {}

def setup_parser(vq_parser):
    # Create the parser
    # vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')

    # Add the arguments
    vq_parser.add_argument("-p",    "--prompts", type=str, help="Text prompts", default=[], dest='prompts')
    vq_parser.add_argument("-sp",   "--spot", type=str, help="Spot Text prompts", default=[], dest='spot_prompts')
    vq_parser.add_argument("-spo",  "--spot_off", type=str, help="Spot off Text prompts", default=[], dest='spot_prompts_off')
    vq_parser.add_argument("-spf",  "--spot_file", type=str, help="Custom spot file", default=None, dest='spot_file')
    vq_parser.add_argument("-l",    "--labels", type=str, help="ImageNet labels", default=[], dest='labels')
    vq_parser.add_argument("-vp",   "--vector_prompts", type=str, help="Vector prompts", default="textoff", dest='vector_prompts')
    vq_parser.add_argument("-ip",   "--image_prompts", type=str, help="Image prompts", default=[], dest='image_prompts')
    vq_parser.add_argument("-ipw",  "--image_prompt_weight", type=float, help="Weight for image prompt", default=None, dest='image_prompt_weight')
    vq_parser.add_argument("-ips",  "--image_prompt_shuffle", type=str2bool, help="Shuffle image prompts", default=False, dest='image_prompt_shuffle')
    vq_parser.add_argument("-il",   "--image_labels", type=str, help="Image prompts", default=None, dest='image_labels')
    vq_parser.add_argument("-ilw",  "--image_label_weight", type=float, help="Weight for image prompt", default=1.0, dest='image_label_weight')
    vq_parser.add_argument("-i",    "--iterations", type=int, help="Number of iterations", default=None, dest='iterations')
    vq_parser.add_argument("-se",   "--save_every", type=int, help="Save image iterations", default=10, dest='save_every')
    vq_parser.add_argument("-de",   "--display_every", type=int, help="Display image iterations", default=20, dest='display_every')
    vq_parser.add_argument("-dc",   "--display_clear", type=str2bool, help="Clear dispaly when updating", default=False, dest='display_clear')
    vq_parser.add_argument("-ove",  "--overlay_every", type=int, help="Overlay image iterations", default=10, dest='overlay_every')
    vq_parser.add_argument("-ovo",  "--overlay_offset", type=int, help="Overlay image iteration offset", default=0, dest='overlay_offset')
    vq_parser.add_argument("-ovu",  "--overlay_until", type=int, help="Last iteration to continue applying overlay image", default=None, dest='overlay_until')
    vq_parser.add_argument("-ovi",  "--overlay_image", type=str, help="Overlay image (if not init)", default=None, dest='overlay_image')
    vq_parser.add_argument(         "--quality", type=str, help="draft, normal, better, best", default="normal", dest='quality')
    vq_parser.add_argument("-asp",  "--aspect", type=str, help="widescreen, square", default="widescreen", dest='aspect')
    vq_parser.add_argument("-ezs",  "--ezsize", type=str, help="small, medium, large", default=None, dest='ezsize')
    vq_parser.add_argument("-sca",  "--scale", type=float, help="scale (instead of ezsize)", default=None, dest='scale')
    vq_parser.add_argument("-ova",  "--overlay_alpha", type=int, help="Overlay alpha (0-255)", default=None, dest='overlay_alpha')    
    vq_parser.add_argument("-s",    "--size", nargs=2, type=int, help="Image size (width height)", default=None, dest='size')
    vq_parser.add_argument("-ii",   "--init_image", type=str, help="Initial image", default=None, dest='init_image')
    vq_parser.add_argument("-iia",  "--init_image_alpha", type=int, help="Init image alpha (0-255)", default=200, dest='init_image_alpha')
    vq_parser.add_argument("-in",   "--init_noise", type=str, help="Initial noise image (pixels or gradient)", default="pixels", dest='init_noise')
    vq_parser.add_argument("-ti",   "--target_images", type=str, help="Target images", default=None, dest='target_images')
    vq_parser.add_argument("-anim", "--animation_dir", type=str, help="Animation output dir", default=None, dest='animation_dir')    
    vq_parser.add_argument("-ana",  "--animation_alpha", type=int, help="Forward blend for consistency", default=128, dest='animation_alpha')
    vq_parser.add_argument("-iw",   "--init_weight", type=float, help="Initial weight (main=spherical)", default=None, dest='init_weight')
    vq_parser.add_argument("-iwd",  "--init_weight_dist", type=float, help="Initial weight dist loss", default=0., dest='init_weight_dist')
    vq_parser.add_argument("-iwc",  "--init_weight_cos", type=float, help="Initial weight cos loss", default=0., dest='init_weight_cos')
    vq_parser.add_argument("-iwp",  "--init_weight_pix", type=float, help="Initial weight pix loss", default=0., dest='init_weight_pix')
    vq_parser.add_argument(         "--perceptors", type=str, help="perceptors (clip/slip/mixed)", default="clip", dest='perceptors')
    vq_parser.add_argument(         "--clip_models", type=str, help="CLIP model", default=None, dest='clip_models')
    vq_parser.add_argument("-nps",  "--noise_prompt_seeds", nargs="*", type=int, help="Noise prompt seeds", default=[], dest='noise_prompt_seeds')
    vq_parser.add_argument("-npw",  "--noise_prompt_weights", nargs="*", type=float, help="Noise prompt weights", default=[], dest='noise_prompt_weights')
    vq_parser.add_argument("-lr",   "--learning_rate", type=float, help="Learning rate", default=0.2, dest='learning_rate')
    vq_parser.add_argument("-lrd",  "--learning_rate_drops", nargs="*", type=float, help="When to drop learning rate (relative to iterations)", default=[75], dest='learning_rate_drops')
    vq_parser.add_argument("-as",   "--auto_stop", type=str2bool, help="Auto stopping", default=False, dest='auto_stop')
    vq_parser.add_argument("-cuts", "--num_cuts", type=int, help="Number of cuts", default=None, dest='num_cuts')
    vq_parser.add_argument("-bats", "--batches", type=int, help="How many batches of cuts", default=None, dest='batches')
    vq_parser.add_argument("-cutp", "--cut_power", type=float, help="Cut power", default=1., dest='cut_pow')
    vq_parser.add_argument("--seed", type=str, help="Seed", default=None, dest='seed')
    vq_parser.add_argument("-opt",  "--optimiser", type=str, help="Optimiser (Adam, AdamW, Adagrad, Adamax, DiffGrad, or AdamP)", default='Adam', dest='optimiser')
    vq_parser.add_argument("-vid",  "--video", type=str2bool, help="Create video frames?", default=False, dest='make_video')
    vq_parser.add_argument("-d",    "--deterministic", type=str2bool, help="Enable cudnn.deterministic?", default=False, dest='cudnn_determinism')
    vq_parser.add_argument("--palette", type=str, help="target palette", default=None, dest='palette')
    vq_parser.add_argument("--transparent", type=str2bool, help="enable transparency", default=False, dest='transparency')
    vq_parser.add_argument("--alpha_weight", type=float, help="strenght of alpha loss", default=1., dest='alpha_weight')
    vq_parser.add_argument("--alpha_use_g", type=str2bool, help="use gaussian mask weighting", default=False, dest='alpha_use_g')
    vq_parser.add_argument("--alpha_gamma", type=float, help="width-relative sigma for the alpha gaussian", default=4., dest='alpha_gamma')

    return vq_parser

def process_args(vq_parser, namespace=None):
    global global_aspect_width
    global cur_iteration, cur_anim_index, anim_output_files, anim_cur_zs, anim_next_zs;
    global global_spot_file, global_given_args
    global best_loss, best_iter, best_z, num_loss_drop, max_loss_drops, iter_drop_delay

    if namespace == None:
        # command line: use ARGV to get args
        args = vq_parser.parse_args()
    elif isnotebook() or hasattr(namespace, 'skip_args'):
        args = vq_parser.parse_args(args=[], namespace=namespace)
    else:
        # sometimes there are both settings and cmd line
        args = vq_parser.parse_args(namespace=namespace)        

    # https://stackoverflow.com/a/66765255/1010653
    global_given_args = {
            opt.dest: getattr(args, opt.dest)
            for opt in vq_parser._option_string_actions.values()
            if hasattr(args, opt.dest) and opt.default != getattr(args, opt.dest)
        }
    # print("NON DEFAULT ARGS ARE")
    # print(global_given_args)
    # sys.exit(0)

    if args.cudnn_determinism:
        torch.backends.cudnn.deterministic = True

    quality_to_clip_models_table = {
        'clip': {
            'draft': 'ViT-B/16',
            'normal': 'ViT-B/32,ViT-B/16',
            'better': 'RN50,ViT-B/32,ViT-B/16',
            'best': 'RN50x4,ViT-B/32,ViT-B/16',
            'supreme': 'RN50x4,RN101,ViT-B/32,ViT-B/16'
        },
        'slip': {
            'draft': 'SLIP_VITB16',
            'normal': 'SLIP_VITB16,SLIP_CC3M',
            'better': 'SLIP_VITB16,SLIP_CC3M,SLIP_CC12M',
            'best': 'SLIP_VITB16,SLIP_CC3M,SLIP_CC12M,SLIP_VITS16',
            'supreme': 'SLIP_VITB16,SLIP_CC3M,SLIP_CC12M,SLIP_VITS16,SLIP_VITL16'
        },
        'mixed': {
            'draft': 'ViT-B/16',
            'normal': 'ViT-B/16,SLIP_VITB16',
            'better': 'RN50,ViT-B/16,SLIP_VITB16',
            'best': 'RN50x4,ViT-B/16,SLIP_VITB16',
            'supreme': 'RN50x4,RN101,ViT-B/16,SLIP_VITB16'
        }
    }

    quality_to_iterations_table = {
        'draft': 200,
        'normal': 250,
        'better': 300,
        'best': 350,
        'supreme': 400
    }
    quality_to_scale_table = {
        'draft': 1,
        'normal': 2,
        'better': 3,
        'best': 4,
        'supreme': 5
    }
    # this should be replaced with logic that does somethings
    # smart based on available memory (eg: size, num_models, etc)
    quality_to_num_cuts_table = {
        'draft': 24,
        'normal': 30,
        'better': 36,
        'best': 12,
        'supreme': 8
    }

    quality_to_batches_table = {
        'draft': 1,
        'normal': 1,
        'better': 1,
        'best': 2,
        'supreme': 4
    }

    if args.quality not in quality_to_clip_models_table[args.perceptors]:
        print("Qualitfy setting not understood, aborting -> ", args.quality)
        exit(1)

    if args.clip_models is None:
        args.clip_models = quality_to_clip_models_table[args.perceptors][args.quality]
    if args.iterations is None:
        args.iterations = quality_to_iterations_table[args.quality]
    if args.num_cuts is None:
        args.num_cuts = quality_to_num_cuts_table[args.quality]
    if args.batches is None:
        args.batches = quality_to_batches_table[args.quality]
    if args.ezsize is None and args.scale is None:
        args.scale = quality_to_scale_table[args.quality]

    size_to_scale_table = {
        'small': 1,
        'medium': 2,
        'large': 4
    }
    aspect_to_size_table = {
        'square': [144, 144],
        'portrait': [128, 160],
        'widescreen': [192, 108]  # vqgan trims to 192x96, 384x208, 576x320, 768x432, 960x528, 1152x640, etc
    }

    # determine size if not set
    if args.size is None:
        size_scale = args.scale
        if size_scale is None:
            if args.ezsize in size_to_scale_table:
                size_scale = size_to_scale_table[args.ezsize]
            else:
                print("EZ Size not understood, aborting -> ", args.ezsize)
                exit(1)
        if args.aspect in aspect_to_size_table:
            base_size = aspect_to_size_table[args.aspect]
            base_width = int(size_scale * base_size[0])
            base_height = int(size_scale * base_size[1])
            args.size = [base_width, base_height]
        elif args.aspect =="retain" and args.init_image is not None:
            img_pil = Image.open(real_glob(args.init_image)[0])
            w,h = img_pil.size
            asp = h/w #w is base
            h = int(144*asp*size_scale)
            w = int(144*size_scale)
            args.size = [w,h]
        else:
            print("aspect not understood, aborting -> ", args.aspect)
            exit(1)

    global_aspect_width = args.size[0] / args.size[1]

    if args.init_noise.lower() == "none":
        args.init_noise = None

    # Split text prompts using the pipe character
    if args.prompts:
        args.prompts = [phrase.strip() for phrase in args.prompts.split("|")]

    # Split text prompts using the pipe character
    if args.target_images:
        args.target_images = [phrase.strip() for phrase in args.target_images.split("|")]

    # Split text prompts using the pipe character
    if args.spot_prompts:
        args.spot_prompts = [phrase.strip() for phrase in args.spot_prompts.split("|")]

    # Split text prompts using the pipe character
    if args.spot_prompts_off:
        args.spot_prompts_off = [phrase.strip() for phrase in args.spot_prompts_off.split("|")]

    # Split text labels using the pipe character
    if args.labels:
        args.labels = [phrase.strip() for phrase in args.labels.split("|")]

    # Split target images using the pipe character
    if args.image_prompts:
        args.image_prompts = real_glob(args.image_prompts)

    # Split text prompts using the pipe character
    if args.vector_prompts:
        if args.vector_prompts.lower() == "none" or args.vector_prompts == "0":
            # print("----> EMPTY VECTOR PROMPT")
            args.vector_prompts = []
        else:
            args.vector_prompts = [phrase.strip() for phrase in args.vector_prompts.split("|")]
    else:
        # print("----> NO VECTOR PROMPT")
        args.vector_prompts = []

    if args.palette is not None:
        args.palette = palette_from_string(args.palette)

    if args.overlay_image is not None and args.overlay_every <= 0:
        args.overlay_image = None

    clip_models = args.clip_models.split(",")
    args.clip_models = [model.strip() for model in clip_models]

    # Make video steps directory
    if args.make_video:
        if not os.path.exists('steps'):
            os.mkdir('steps')

    if args.learning_rate_drops is None:
        args.learning_rate_drops = []
    else:
        args.learning_rate_drops = [int(map_number(n, 0, 100, 0, args.iterations-1)) for n in args.learning_rate_drops]

    # reset global animation variables
    cur_iteration=0
    best_iter = cur_iteration
    best_loss = 1e20
    num_loss_drop = 0
    max_loss_drops = len(args.learning_rate_drops)
    iter_drop_delay = 12
    best_z = None

    cur_anim_index=None
    anim_output_files=[]
    anim_cur_zs=[]
    anim_next_zs=[]

    global_spot_file = args.spot_file

    return args

def reset_settings():
    global global_pixray_settings
    global_pixray_settings = {}

def add_settings(**kwargs):
    global global_pixray_settings
    for k, v in kwargs.items():
        # TODO: is None / "None" a special case or not?
        global_pixray_settings[k] = v
        # if v is None or v == "None":
        #     # just remove the key if it is there
        #     global_pixray_settings.pop(k, None)
        # else:

def get_settings():
    global global_pixray_settings
    return global_pixray_settings.copy()


def parse_known_args_with_optional_yaml(parser, namespace=None):
    parser.add_argument(
        '--config_file',
        dest='config_file',
        type=argparse.FileType(mode='r'))
    
    arguments, unknown = parser.parse_known_args(namespace=namespace)
    if arguments.config_file:
        data = yaml.load(arguments.config_file, Loader=yaml.SafeLoader)
        delattr(arguments, 'config_file')
        arg_dict = arguments.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                if(key not in arg_dict or arg_dict[key] is None):
                    arg_dict[key] = []
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    
    return arguments, unknown

def initialize_logging(settings_core):
    if settings_core.output_dir is not None and settings_core.output_dir.strip() != '':
        logfile = get_file_path(settings_core.output_dir, settings_core.output, '.log')
        logging.basicConfig(level=logging.DEBUG, filename=logfile, filemode='w+')

def apply_settings():
    global global_pixray_settings
    settingsDict = None

    # first pass - only add things here that can trigger other parser additions (drawers, filters, losses)
    # Create the parser
    vq_parser = argparse.ArgumentParser(description='Image generation using VQGAN+CLIP')
    vq_parser.add_argument("--drawer",  type=str, help="clipdraw, pixel, etc", default="vqgan", dest='drawer')
    vq_parser.add_argument("--filters", type=str, help="Image Filtering", default=None, dest='filters')
    vq_parser.add_argument("--losses", "--custom_loss", type=str, help="implement a custom loss type through LossInterface. example: edge", default=None, dest='custom_loss')
    vq_parser.add_argument("-o",    "--output", type=str, help="Output filename", default="output.png", dest='output')
    vq_parser.add_argument("-od", "--output_dir", type=str, help="Output file directory", default="", dest='output_dir')
    
    settingsDict = SimpleNamespace(**global_pixray_settings)
    settings_core, unknown = parse_known_args_with_optional_yaml(vq_parser, namespace=settingsDict)

    initialize_logging(settings_core)
    vq_parser = setup_parser(vq_parser)
    class_table[settings_core.drawer].add_settings(vq_parser)

    if settings_core.filters is not None:
        # probably should DRY but...
        filts = settings_core.filters.split(",")
        filts = [f.strip() for f in filts]
        for f in filts:
            f = f.split(':')[0]
            filters_class_table[f].add_settings(vq_parser)

    if settings_core.custom_loss is not None:
        # probably should DRY but...
        custom_losses = settings_core.custom_loss.split(",")
        custom_losses = [loss.strip() for loss in custom_losses]
        for l in custom_losses:
            l = l.split(':')[0]
            loss_class_table[l].add_settings(vq_parser)

    if len(global_pixray_settings) > 0:
        # check for any bogus entries in the settings
        dests = [d.dest for d in vq_parser._actions]
        for k in global_pixray_settings:
            if not k in dests and k != "skip_args":
                raise ValueError(f"Requested setting not found, aborting: {k}={global_pixray_settings[k]}")

        # convert dictionary to easyDict
        # which can be used as an argparse namespace instead
        # settingsDict = easydict.EasyDict(global_pixray_settings)
        settingsDict = SimpleNamespace(**global_pixray_settings)

    settings = process_args(vq_parser, settingsDict)
    logging.debug(json.dumps(settings, default=lambda o: o.__dict__, sort_keys=True, indent=4))
    return settings

def add_custom_loss(name, customloss):
    # unfortunately this way it cannot access globals when initializing 
    assert issubclass(customloss,LossInterface)
    loss_class_table.update({
        name: customloss,
    })

def command_line_override():
    global global_pixray_settings
    settingsDict = None
    vq_parser = setup_parser()
    settings = process_args(vq_parser)
    return settings

def main():
    settings = apply_settings()
    print(f"Running with {settings.num_cuts}x{settings.batches} = {settings.num_cuts*settings.batches} cuts")
    do_init(settings)
    do_run(settings)
    # global drawer
    # drawer.to_svg()

if __name__ == '__main__':
    main()
