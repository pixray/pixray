# code adapted from https://github.com/facebookresearch/SLIP/issues/2#issuecomment-1001052198

import sys
import os
from collections import OrderedDict

import torch 
import torch.nn as nn
from torchvision import transforms

from clip import clip

all_slip_models =  ["SLIP_VITS16", "SLIP_VITB16", "SLIP_VITL16",
                    "SIMCLR_VITS16",
                    "CLIP_VITS16", "CLIP_VITB16", "CLIP_VITL16"]


from util import wget_file

def normalize(img, input_range = None):
    if input_range is None:
        minv = img.min()
    else:
        minv = input_range[0]
    img = img - minv

    if input_range is None:
        maxv = img.max()
    else:
        maxv = input_range[1] - minv

    if maxv != 0:
        img = img / maxv

    return img

def adjust_range(img, out_range, input_range = None):
    img = normalize(img, input_range = input_range)
    img = img * (out_range[1] - out_range[0])
    img = img + out_range[0]
    return img

class CLIP_Base():
    # Default CLIP model from OpenAI
    def __init__(self, model, preprocess, device):
        self.device = device
        self.model  = model.eval()
        self.input_resolution = self.model.visual.input_resolution
        self.output_dim = self.model.visual.output_dim

        self.preprocess_transform = transforms.Compose([
            transforms.Resize(self.input_resolution),
            transforms.CenterCrop(self.input_resolution),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])

    def preprocess(self, imgs, input_range = None):
        imgs = adjust_range(imgs, [0.,1.], input_range = input_range)
        return self.preprocess_transform(imgs)

    def encode_image(self, imgs, input_range = None, apply_preprocess = True):
        if apply_preprocess:
            imgs = self.preprocess(imgs, input_range = None)
        img_embeddings = self.model.encode_image(imgs)
        return img_embeddings / img_embeddings.norm(dim=-1, keepdim=True)

    def encode_text(self, text):
        text = clip.tokenize(text).to(self.device)
        return self.model.encode_text(text).float()

    def encode_texts(self, texts):
        text_embeddings = torch.stack([self.model.encode_text(clip.tokenize(text).to(self.device)).detach().clone() for text in texts])
        return text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

# TODO: this is very hacky, must fix this later (submodule dependency)
SLIP_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SLIP')
# print("APPENDING PATH ", SLIP_PATH)
sys.path.append(SLIP_PATH)
import models
from tokenizer import SimpleTokenizer
import utils

class SLIP_Base():
    def __init__(self, model_name, device):
        self.device = device
        self.input_resolution = 224

        if model_name == "SLIP_VITS16":
            ckpt_file = f"slip_small_100ep.pt"
        elif model_name == "SLIP_VITB16":
            ckpt_file  = f"slip_base_100ep.pt"
        elif model_name == "SLIP_VITL16":
            ckpt_file = f"slip_large_100ep.pt"
        elif model_name == "SIMCLR_VITS16":
            ckpt_file = f"simclr_small_25ep.pt"
        elif model_name == "CLIP_VITS16":
            ckpt_file = f"clip_small_25ep.pt"
        elif model_name == "CLIP_VITB16":
            ckpt_file = f"clip_base_25ep.pt"
        elif model_name == "CLIP_VITL16":
            ckpt_file = f"clip_large_25ep.pt"
        else:
            print(f"slip model {model_name} not known, aborting")
            sys.exit(1)

        ckpt_path = f"models/{ckpt_file}"
        if not os.path.exists(ckpt_path):
            url = f"https://dl.fbaipublicfiles.com/slip/{ckpt_file}"
            wget_file(url, ckpt_path)

        self.preprocess_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        self.tokenizer = SimpleTokenizer()

        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        # create model
        old_args = ckpt['args']
        old_args.model = model_name

        model = getattr(models, old_args.model)(rand_embed=False,
            ssl_mlp_dim=old_args.ssl_mlp_dim, ssl_emb_dim=old_args.ssl_emb_dim)
        model.cuda().requires_grad_(False).eval()
        model.load_state_dict(state_dict, strict=True)

        n_params = sum(p.numel() for p in model.parameters())
        print("Loaded perceptor %s: %.2fM params" %(model_name, (n_params/1000000)))
        
        self.model = utils.get_model(model)

    def preprocess(self, imgs, input_range = None):
        imgs = adjust_range(imgs, [0.,1.], input_range = input_range)
        return self.preprocess_transform(imgs)

    def encode_image(self, imgs, input_range = None, apply_preprocess = True):
        if apply_preprocess:
            imgs = self.preprocess(imgs, input_range = input_range)

        image_features = self.model.encode_image(imgs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def encode_text(self, texts):
        texts = self.tokenizer(texts).cuda(non_blocking=True)
        texts = texts.view(-1, 77).contiguous()
        text_embeddings = self.model.encode_text(texts)
        return text_embeddings

    def encode_texts(self, texts):
        texts = self.tokenizer(texts).cuda(non_blocking=True)
        texts = texts.view(-1, 77).contiguous()
        text_embeddings = self.model.encode_text(texts)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings.unsqueeze(1)
        

def get_clip_perceptor(clip_model_name, device):
    if clip_model_name in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']:
        perceptor, preprocess = clip.load(clip_model_name, download_root="models")
        perceptor = perceptor.requires_grad_(False).eval().to(device)

        n_params = sum(p.numel() for p in perceptor.parameters())
        print("Loaded CLIP %s: %.2fM params" %(clip_model_name, (n_params/1000000)))
        clip_perceptor = CLIP_Base(perceptor, preprocess, device)

    else:
        clip_perceptor = SLIP_Base(clip_model_name, device)

    return clip_perceptor