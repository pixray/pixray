import torch
from torch import nn, optim
from PIL import Image
from util import *
from urllib.request import urlopen
import torchvision.transforms.functional as TF



from Losses.LossInterface import LossInterface

class EdgeLoss(LossInterface):
    def __init__(self,**kwargs):
        self.image = None
        self.resized = None
        super().__init__(**kwargs)
    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--edge_thickness", nargs=4, type=int, help="this is for the thickness of the edge area (left, right, up, down) 0-pixel size", default=(10,10,10,10), dest='edge_thickness')
        parser.add_argument("--edge_color", nargs=3, type=float, help="this is the color of the specified region, in (R,G,B) 0-255", default=(255,255,255), dest='edge_color')
        parser.add_argument("--edge_color_weight", type=float, help="how much edge color is enforced", default=2, dest='edge_color_weight')
        parser.add_argument("--global_color_weight", type=float, help="how much global color is enforced ", default=0.5, dest='global_color_weight')
        parser.add_argument("--edge_input_image", type=str, help="TBD", default="", dest='edge_input_image')
        return parser

    def parse_settings(self,args):
        #do stuff with args here
        if type(args.edge_color)==str:
            args.edge_color = (255,255,255)
        if args.edge_input_image:
            # now we might overlay an init image
            filelist = None
            if 'http' in args.edge_input_image:
                self.image = [Image.open(urlopen(args.edge_input_image))]
            else:
                filelist = real_glob(args.edge_input_image)
                self.image = [Image.open(f) for f in filelist]
            self.image = self.image[0]
        return args
    
    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        if self.resized is None:
            self.resized = TF.to_tensor(self.image).to(self.device).unsqueeze(0)
            self.resized = TF.resize(self.resized, out.size()[2:4],TF.InterpolationMode.BICUBIC)
        if not args.edge_input_image:
            zers = torch.zeros(out.size()).to(self.device)
            Rval,Gval,Bval = args.edge_color
            zers[:,0,:,:] = Rval/255
            zers[:,1,:,:] = Gval/255
            zers[:,2,:,:] = Bval/255
        else: # args.edge_input_image:
            zers = self.resized
        lmax = out.size()[2]
        rmax = out.size()[3]
        mseloss = nn.MSELoss()
        left, right, upper, lower = args.edge_thickness
        cur_loss = torch.tensor(0.0).cuda()
        lloss = mseloss(out[:,:,:,:left], zers[:,:,:,:left]) 
        rloss = mseloss(out[:,:,:,rmax-right:], zers[:,:,:,rmax-right:])
        uloss = mseloss(out[:,:,:upper,left:rmax-right], zers[:,:,:upper,left:rmax-right]) 
        dloss = mseloss(out[:,:,lmax-lower:,left:rmax-right], zers[:,:,lmax-lower:,left:rmax-right]) 
        if left!=0:
            cur_loss+=lloss
        if right!=0:
            cur_loss+=rloss
        if upper!=0:
            cur_loss+=uloss
        if lower!=0:
            cur_loss+=dloss
        if args.global_color_weight:
            gloss = mseloss(out[:,:,:,:], zers[:,:,:,:]) * args.global_color_weight
            cur_loss+=gloss
        cur_loss *= args.edge_color_weight
        return cur_loss
