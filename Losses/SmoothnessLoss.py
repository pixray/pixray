import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

from Losses.LossInterface import LossInterface

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


class SmoothnessLoss(LossInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--smoothness_weight", type=float, help="strength of smoothness loss effect", default=1, dest='smoothness_weight')
        parser.add_argument("--smoothness_type", type=str, help="enforce smoothness type: default/clipped/log", default='default', dest='smoothness_type')
        parser.add_argument("--smoothness_gaussian_kernel", type=float, help="enforce smoothness aux gaussian blur kernel", default=0, dest='smoothness_gaussian_kernel')
        parser.add_argument("--smoothness_gaussian_std", type=float, help="enforce smoothness aux gaussian blur std", default=1, dest='smoothness_gaussian_std')
        return parser
    
    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        cur_loss = []
        for _,cutouts in cur_cutouts.items():
            if args.smoothness_gaussian_kernel:
                smoothing = GaussianSmoothing(3, args.smoothness_gaussian_kernel, args.smoothness_gaussian_std).to(self.device)
                cutouts = smoothing(cutouts)

            _pixels = cutouts.permute(0,2,3,1).reshape(-1,cutouts.shape[2],3)
            gyr, gxr = torch.gradient(_pixels[:,:,0])
            gyg, gxg = torch.gradient(_pixels[:,:,1])
            gyb, gxb = torch.gradient(_pixels[:,:,2])
            sharpness = torch.sqrt(gyr**2 + gxr**2+ gyg**2 + gxg**2 + gyb**2 + gxb**2)
            if args.smoothness_type=='clipped':
                sharpness = torch.clamp( sharpness, max=0.5 )
            elif args.smoothness_type=='log':
                sharpness = torch.log( torch.ones_like(sharpness)+sharpness )
            sharpness = torch.mean( sharpness )
            cur_loss.append(sharpness*args.smoothness_weight)

        return cur_loss
