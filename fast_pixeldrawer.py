import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from DrawingInterface import DrawingInterface


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


class FastPixelDrawer(DrawingInterface):
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--pixel_size", nargs=2, type=int, help="Pixel size (width height)", default=None, dest='pixel_size')
        parser.add_argument("--pixel_scale", type=float, help="Pixel scale", default=None, dest='pixel_scale')
        return parser

    def __init__(self, settings):
        super(DrawingInterface, self).__init__()

        # NOTE: much of the init logic below duplicated from pixeldrawer.py

        # current logic: assume 16x9, or 4x5, but check for 1x1 (all others must be provided explicitly)
        self.canvas_width = settings.size[0]
        self.canvas_height = settings.size[1]
        if settings.pixel_size is not None:
            self.num_cols, self.num_rows = settings.pixel_size
        elif self.canvas_width == self.canvas_height:
            self.num_cols, self.num_rows = [40, 40]
        elif self.canvas_width < self.canvas_height:
            self.num_cols, self.num_rows = [40, 50]
        else:
            self.num_cols, self.num_rows = [80, 45]

        # we can also "scale" pixels -- scaling "up" meaning fewer rows/cols, etc.
        if settings.pixel_scale is not None and settings.pixel_scale > 0:
            self.num_cols = int(self.num_cols / settings.pixel_scale)
            self.num_rows = int(self.num_rows / settings.pixel_scale)

        shrink = False
        if self.num_cols>self.canvas_width:
            shrink = True
            self.num_cols = self.canvas_width
        if self.num_rows>self.canvas_height:
            shrink = True
            self.num_rows = self.canvas_height
        if shrink:
            print('pixel grid size should not be larger than output pixel size: reducing pixel grid')

        print(f"Running fast pixeldrawer with {self.num_cols}x{self.num_rows} grid")

        self.pixel_size = tuple([self.num_rows, self.num_cols])
        self.output_size = tuple(reversed(settings.size))

    def load_model(self, settings, device):
        pass

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
        return F.interpolate((ref_tensor + 1) / 2, size=self.pixel_size, mode="bilinear", align_corners=False)

    def get_num_resolutions(self):
        return None

    def synth(self, cur_iteration):
        output = F.interpolate(self.z, size=self.output_size, mode="nearest")
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
