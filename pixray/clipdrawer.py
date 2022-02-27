# this is derived from ClipDraw code
# CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders
# Kevin Frans, L.B. Soros, Olaf Witkowski
# https://arxiv.org/abs/2106.14843

from DrawingInterface import DrawingInterface

import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import torchvision
import torchvision.transforms as transforms
import numpy as np
import PIL.Image

class ClipDrawer(DrawingInterface):
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--strokes", type=int, help="number strokes", default=1024, dest='strokes')
        parser.add_argument("--min_stroke_width", type=float, help="min width (percent of height)", default=1, dest='min_stroke_width')
        parser.add_argument("--max_stroke_width", type=float, help="max width (percent of height)", default=5, dest='max_stroke_width')
        return parser

    def __init__(self, settings):
        super(DrawingInterface, self).__init__()

        self.canvas_width = settings.size[0]
        self.canvas_height = settings.size[1]
        self.num_paths = settings.strokes

    def load_model(self, settings, device):
        # Use GPU if available
        pydiffvg.set_use_gpu(torch.cuda.is_available())
        device = torch.device('cuda')
        pydiffvg.set_device(device)

        canvas_width, canvas_height = self.canvas_width, self.canvas_height
        num_paths = self.num_paths
        max_width = settings.max_stroke_width * canvas_height / 100
        min_width = settings.min_stroke_width * canvas_height / 100

        # Initialize Random Curves
        shapes = []
        shape_groups = []
        for i in range(num_paths):
            num_segments = random.randint(1, 3)
            num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.1
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(num_control_points = num_control_points, points = points, stroke_width = torch.tensor((min_width + max_width)/4), is_closed = False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]), fill_color = None, stroke_color = torch.tensor([random.random(), random.random(), random.random(), random.random()]))
            shape_groups.append(path_group)

        # Just some diffvg setup
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

        points_vars = []
        stroke_width_vars = []
        color_vars = []
        for path in shapes:
            path.points.requires_grad = True
            points_vars.append(path.points)
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
        for group in shape_groups:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)

        self.points_vars = points_vars
        self.stroke_width_vars = stroke_width_vars
        self.color_vars = color_vars
        self.img = img
        self.shapes = shapes 
        self.shape_groups  = shape_groups
        self.max_width = max_width
        self.min_width = min_width
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

    def get_opts(self, decay_divisor):
        # Optimizers
        points_optim = torch.optim.Adam(self.points_vars, lr=1.0/decay_divisor)
        width_optim = torch.optim.Adam(self.stroke_width_vars, lr=0.1/decay_divisor)
        color_optim = torch.optim.Adam(self.color_vars, lr=0.01/decay_divisor)
        opts = [points_optim, width_optim, color_optim]
        return opts

    def rand_init(self, toksX, toksY):
        # TODO
        pass

    def init_from_tensor(self, init_tensor):
        # TODO
        pass

    def reapply_from_tensor(self, new_tensor):
        # TODO
        pass

    def get_z_from_tensor(self, ref_tensor):
        return None

    def get_num_resolutions(self):
        return None

    def synth(self, cur_iteration):
        render = pydiffvg.RenderFunction.apply
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        img = render(self.canvas_width, self.canvas_height, 2, 2, cur_iteration, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        self.img = img
        return img

    @torch.no_grad()
    def to_image(self):
        img = self.img.detach().cpu().numpy()[0]
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        img = np.uint8(img * 254)
        # img = np.repeat(img, 4, axis=0)
        # img = np.repeat(img, 4, axis=1)
        pimg = PIL.Image.fromarray(img, mode="RGB")
        return pimg

    def clip_z(self):
        with torch.no_grad():
            for path in self.shapes:
                path.stroke_width.data.clamp_(self.min_width, self.max_width)
            for group in self.shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)

    def get_z(self):
        return None

    def get_z_copy(self):
        return None

    def set_z(self, new_z):
        return None

### EXTERNAL INTERFACE
### load_vqgan_model

if __name__ == '__main__':
    main()
