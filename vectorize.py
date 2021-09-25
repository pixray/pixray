import argparse
import sys
import json
import numpy as np
from sklearn import metrics
from sklearn import svm
import os
from tqdm import tqdm
from util import real_glob

import torch
from CLIP import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

perceptors = {}

def init(args):
    global perceptors, resolutions

    model, preprocess = clip.load("ViT-B/32")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    jit = True if float(torch.__version__[:3]) < 1.8 else False

    models = args.models.split(",")
    args.models = [model.strip() for model in models]

    for clip_model in args.models:
        model, preprocess = clip.load(clip_model, jit=jit)
        perceptor = model.eval().requires_grad_(False).to(device)
        perceptors[clip_model] = perceptor

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

def vector_to_json_array(v):
    return json.dumps(v.tolist())

def spew_vectors(args, inputs, outfile):
    global perceptors, resolutions

    input_files = real_glob(inputs)
    save_table = {}
    for clip_model in args.models:
        perceptor = perceptors[clip_model]
        input_resolution = perceptor.visual.input_resolution
        print(f"Running {clip_model} at {input_resolution}")
        preprocess = Compose([
            Resize(input_resolution, interpolation=Image.BICUBIC),
            CenterCrop(input_resolution),
            ToTensor()
        ])
        image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
        image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

        images = fetch_images(preprocess, input_files);

        features = do_image_features(perceptor, images, image_mean, image_std)
        print(f"saving {features.shape} to {clip_model}")
        save_table[clip_model] = vector_to_json_array(features)

    with open(outfile, 'w') as fp:
        json.dump(save_table, fp)

def main():
    parser = argparse.ArgumentParser(description="Do vectory things")
    parser.add_argument("--models", type=str, help="CLIP model", default='ViT-B/32,ViT-B/16,RN50,RN50x4', dest='models')
    parser.add_argument("--inputs", type=str, help="Images to process", default=None, dest='inputs')
    parser.add_argument("--z-dim", dest='z_dim', type=int, default=100,
                        help="z dimension of vectors")
    parser.add_argument("--encoded-vectors", type=str, default=None,
                        help="Comma separated list of json arrays")
    parser.add_argument("--encoded-true", type=str, default=None,
                        help="Comma separated list of json arrays (true)")
    parser.add_argument("--encoded-false", type=str, default=None,
                        help="Comma separated list of json arrays (false)")
    parser.add_argument('--thresh', dest='thresh', default=False, action='store_true',
                        help="Compute thresholds for attribute vectors classifiers")
    parser.add_argument('--svm', dest='svm', default=False, action='store_true',
                        help="Use SVM for computing attribute vectors")
    parser.add_argument("--limit", dest='limit', type=int, default=None,
                        help="Limit number of inputs when computing atvecs")
    parser.add_argument("--attribute-vectors", dest='attribute_vectors', default=None,
                        help="use json file as source of attribute vectors")
    parser.add_argument("--attribute-thresholds", dest='attribute_thresholds', default=None,
                        help="use these non-zero values for binary classifier thresholds")
    parser.add_argument("--attribute-set", dest='attribute_set', default="all",
                        help="score ROC/accuracy against true/false/all")
    parser.add_argument('--attribute-indices', dest='attribute_indices', default=None, type=str,
                        help="indices to select specific attribute vectors")
    parser.add_argument("--avg-diff", dest='avg_diff', type=str, default=None,
                        help="Two lists of vectors to average and then diff")
    parser.add_argument("--svm-diff", dest='svm_diff', type=str, default=None,
                        help="Two lists of vectors to average and then svm diff")
    parser.add_argument('--outfile', dest='outfile', default=None,
                        help="Output json file for vectors.")
    args = parser.parse_args()

    init(args)
    spew_vectors(args, args.inputs, args.outfile)

if __name__ == '__main__':
    main()