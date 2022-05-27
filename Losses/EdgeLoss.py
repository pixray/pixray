import torch
from torch import nn

from util import get_single_rgb
from PIL import Image
from util import *
from urllib.request import urlopen
import torchvision.transforms.functional as TF


from Losses.LossInterface import LossInterface

class EdgeLoss(LossInterface):
    def __init__(self,**kwargs):
        self.image = None
        self.resized = None
        self.resized_mask = None
        super().__init__(**kwargs)
    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--edge_thickness", type=int, help="thickness of the edge area all the way around (percent)", default=5, dest='edge_thickness')
        parser.add_argument("--edge_margins", nargs=4, type=int, help="this is for the thickness of each edge (left, right, up, down) 0-pixel size", default=None, dest='edge_margins')
        parser.add_argument("--edge_color", type=str, help="this is the color of the specified region", default="white", dest='edge_color')
        parser.add_argument("--edge_color_weight", type=float, help="how much edge color is enforced", default=0.1, dest='edge_color_weight')
        parser.add_argument("--global_color_weight", type=float, help="how much global color is enforced ", default=0.05, dest='global_color_weight')
        parser.add_argument("--edge_input_image", type=str, help="TBD", default="", dest='edge_input_image')
        parser.add_argument("--edge_mask_image", type=str, help="TBD", default="", dest='edge_mask_image')
        return parser

    def parse_settings(self,args):
        #do stuff with args here
        if type(args.edge_color)==str:
            args.edge_color = get_single_rgb(args.edge_color)
        if args.edge_margins is None:
            t = args.edge_thickness
            args.edge_margins = (t, t, t, t)
        if args.edge_input_image:
            # now we might overlay an init image
            filelist = None
            if 'http' in args.edge_input_image:
                self.image = [Image.open(urlopen(args.edge_input_image))]
            else:
                filelist = real_glob(args.edge_input_image)
                self.image = [Image.open(f) for f in filelist]
            self.image = self.image[0].convert('RGB')
        if args.edge_mask_image:
            # now we might overlay an init image
            filelist = None
            if 'http' in args.edge_mask_image:
                self.mask = [Image.open(urlopen(args.edge_mask_image))]
            else:
                filelist = real_glob(args.edge_mask_image)
                self.mask = [Image.open(f) for f in filelist]
            self.mask = self.mask[0].convert('L')
        else:
            self.mask = None
        return args
    
    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        if self.resized is None and self.image is not None:
            self.resized = TF.to_tensor(self.image).to(self.device).unsqueeze(0)
            self.resized = TF.resize(self.resized, out.size()[2:4],TF.InterpolationMode.BICUBIC)
            print("IMAGE MASK: ", self.resized.shape)
        if self.resized_mask is None and self.mask is not None:
            self.resized_mask = TF.to_tensor(self.mask).to(self.device).unsqueeze(0)
            self.resized_mask = TF.resize(self.resized_mask, out.size()[2:4],TF.InterpolationMode.BICUBIC)
            print("RESIZED MASK: ", self.resized_mask.shape)
        if not self.image:
            zers = torch.zeros(out.size()).to(self.device)
            Rval,Gval,Bval = args.edge_color
            zers[:,0,:,:] = Rval
            zers[:,1,:,:] = Gval
            zers[:,2,:,:] = Bval
        else: # args.edge_input_image:
            zers = self.resized
        cur_loss = torch.tensor(0.0).cuda()
        mseloss = nn.MSELoss()
        if not self.mask:
            lmax = out.size()[2]
            rmax = out.size()[3]
            left, right, upper, lower = args.edge_margins
            left = int(map_number(left, 0, 100, 0, rmax))
            right = int(map_number(right, 0, 100, 0, rmax))
            upper = int(map_number(upper, 0, 100, 0, lmax))
            lower = int(map_number(lower, 0, 100, 0, lmax))
            # print(left, right, upper, lower)
            lloss = mseloss(out[:,:,:,:left], zers[:,:,:,:left]) 
            rloss = mseloss(out[:,:,:,rmax-right:], zers[:,:,:,rmax-right:])
            uloss = mseloss(out[:,:,:upper,left:rmax-right], zers[:,:,:upper,left:rmax-right]) 
            dloss = mseloss(out[:,:,lmax-lower:,left:rmax-right], zers[:,:,lmax-lower:,left:rmax-right]) 
            if left!=0:
                cur_loss += lloss
            if right!=0:
                cur_loss += rloss
            if upper!=0:
                cur_loss += uloss
            if lower!=0:
                cur_loss += dloss
        else:
            maked_out = torch.where(self.resized_mask > 0, zers, out)
            mask_loss = mseloss(maked_out, zers)
            cur_loss += mask_loss
        if args.global_color_weight:
            gloss = mseloss(out[:,:,:,:], zers[:,:,:,:]) * args.global_color_weight
            cur_loss+=gloss
        cur_loss *= args.edge_color_weight
        return cur_loss
