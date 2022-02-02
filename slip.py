# code adapted from https://github.com/facebookresearch/SLIP/issues/2#issuecomment-1001052198

import sys
import os
from collections import OrderedDict

import torch 
import torch.nn as nn
from torchvision import transforms

from clip import clip

all_slip_models =  ["SLIP_VITS16", "SLIP_VITB16", "SLIP_VITL16",
                    "SLIP_CC3M", "SLIP_CC12M",
                    "SIMCLR_VITS16",
                    "CLIP_VITS16", "CLIP_VITB16", "CLIP_VITL16"]

all_blip_models = {"BLIP_BASE": "model_base.pth",
                   "BLIP_BASE_14M": "model_base_14M.pth",
                   "BLIP_LARGE": "model_large.pth"}

blip_vit_types  = {"BLIP_BASE": "base",
                   "BLIP_BASE_14M": "base",
                   "BLIP_LARGE": "large"}


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
        text_embedding = self.model.encode_text(text).float()
        return text_embedding

    def encode_texts(self, texts):
        text_embeddings = torch.stack([self.model.encode_text(clip.tokenize(text).to(self.device)).detach().clone() for text in texts])
        te_normed = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        return te_normed

#TODO: cant enable SLIP and BLIP imports
# TODO: this is very hacky, must fix this later (submodule dependency)
# SLIP_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SLIP')
# # print("APPENDING PATH ", SLIP_PATH)
# sys.path.append(SLIP_PATH)
# import models
# from tokenizer import SimpleTokenizer
# import utils

class SLIP_Base():
    def __init__(self, model_name, device):
        self.device = device
        self.input_resolution = 224

        # HA HA HA, this could be a lookup table but I'm too lazy to change it
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
        elif model_name == "SLIP_CC3M":
            ckpt_file  = f"slip_base_cc3m_40ep.pt"
        elif model_name == "SLIP_CC12M":
            ckpt_file  = f"slip_base_cc12m_35ep.pt"
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
        # these two are the same model on different training data...
        if old_args.model == "SLIP_CC3M" or old_args.model == "SLIP_CC12M":
            old_args.model = "SLIP_VITB16"

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
        
BLIP_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'BLIP')
# print("APPENDING PATH ", SLIP_PATH)
sys.path.append(BLIP_PATH)
from models.blip import blip_feature_extractor
from collections import namedtuple

class BLIP_Base():
    def __init__(self, model_name, device):
        self.device = device
        FakeImage = namedtuple('FakeImage', 'device')
        self.fake_image = FakeImage(device=device)

        self.input_resolution = 224

        if model_name in all_blip_models:
            ckpt_file = all_blip_models[model_name]
        else:
            print(f"blip model {model_name} not known, aborting")
            sys.exit(1)

        ckpt_path = f"models/blip_{ckpt_file}"
        url = f"https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/{ckpt_file}"
        #TODO: model should load ckpt_path not url
        # if not os.path.exists(ckpt_path):
        #     wget_file(url, ckpt_path)

        self.preprocess_transform = transforms.Compose([
                transforms.Resize((self.input_resolution,self.input_resolution),interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ])

        model = blip_feature_extractor(pretrained=url, image_size=224, vit=blip_vit_types[model_name])
        # model.eval()
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print("Loaded perceptor %s: %.2fM params" %(model_name, (n_params/1000000)))

        self.model = model

    def preprocess(self, imgs, input_range = None):
        imgs = adjust_range(imgs, [0.,1.], input_range = input_range)
        return self.preprocess_transform(imgs)

    def encode_image(self, imgs, input_range = None, apply_preprocess = True):
        if apply_preprocess:
            imgs = self.preprocess(imgs, input_range = input_range)

        image_features = self.model(imgs, '', mode='image')
        # print("IF", image_features.shape)
        dim = image_features.shape[2]
        image_features = image_features[0][0].reshape(1, dim)
        return image_features

    def encode_text(self, texts):
        text_features = self.model(self.fake_image, texts, mode='text')
        # print("TF", text_features.shape)
        dim = text_features.shape[2]
        text_features = text_features[0][0].detach().clone().reshape(1, dim)
        return text_features

    def encode_texts(self, texts):
        print("THIS IS NEVER CALLED?")
        sys.exit(1)
        text_features = self.model(self.fake_image, texts, mode='text')
        text_features = text_features[0][0].reshape(1, 768)
        return text_features.unsqueeze(1)
        

def get_clip_perceptor(clip_model_name, device):
    if clip_model_name in ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14']:
        perceptor, preprocess = clip.load(clip_model_name, download_root="models")
        perceptor = perceptor.requires_grad_(False).eval().to(device)

        n_params = sum(p.numel() for p in perceptor.parameters())
        in_res = perceptor.visual.input_resolution
        print(f"Loaded CLIP {clip_model_name}: {in_res}x{in_res} and {n_params/1000000:.2f}M params")
        clip_perceptor = CLIP_Base(perceptor, preprocess, device)
    elif clip_model_name in all_blip_models:
        clip_perceptor = BLIP_Base(clip_model_name, device)
    else:
        clip_perceptor = SLIP_Base(clip_model_name, device)

    return clip_perceptor