import torch
from torch import nn, optim




from LossInterface import LossInterface

class EdgeLoss(LossInterface):
    def __init__(self,custom_init,**kwargs):
        print(kwargs)
        print(custom_init,"custom init 0 message :)")
        super().__init__(**kwargs)
    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("-ec",  "--use_edge_color", type=bool, help="enabling this allows controlling colors of reigons around edges", default=False, dest='use_edge_color')
        parser.add_argument("-ed",  "--edge_thicness", nargs=4, type=int, help="this is for the thickness of the edge area (left, right, up, down) 0-pixel size", default=(10,10,10,10), dest='edge_thickness')
        parser.add_argument("-ecol",  "--edge_color", nargs=3, type=float, help="this is the color of the specified region, in (R,G,B) 0-255", default=(255,255,255), dest='edge_color')
        parser.add_argument("-ecw",  "--edge_color_weight", type=float, help="how much edge color is enforced", default=2, dest='edge_color_weight')
        parser.add_argument("-gc",  "--use_global_color", type=bool, help="if you want a tendency for globally to be this color", default=False, dest='use_global_color')
        parser.add_argument("-gcw",  "--global_color_weight", type=float, help="how much global color is enforced ", default=0.5, dest='global_color_weight')
        return parser

    def parse_settings(self,args):
        #do stuff with args here
        if type(args.edge_color)==str:
            args.edge_color = (255,255,255)
        return args
    
    def forward(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        zers = torch.zeros(out.size()).cuda()
        # print(out.size())
        # print(out.size()[1])
        lmax = out.size()[2]
        rmax = out.size()[3]
        # print(rmax,lmax)
        # r,g,b
        Rval,Gval,Bval = args.edge_color
        zers[:,0,:,:] = Rval/255
        zers[:,1,:,:] = Gval/255
        zers[:,2,:,:] = Bval/255
        # print(zers)
        mseloss = nn.MSELoss()
        left, right, upper, lower = args.edge_thickness
        cur_loss = torch.tensor(0.0).cuda()
        lloss = mseloss(out[:,:,:,:left], zers[:,:,:,:left]) 
        rloss = mseloss(out[:,:,:,rmax-right:], zers[:,:,:,rmax-right:])
        uloss = mseloss(out[:,:,:upper,left:rmax-right], zers[:,:,:upper,left:rmax-right]) 
        dloss = mseloss(out[:,:,lmax-lower:,left:rmax-right], zers[:,:,lmax-lower:,left:rmax-right]) 
        gloss = mseloss(out[:,:,:,:], zers[:,:,:,:]) * args.global_color_weight
        # print(lloss, rloss, uloss, dloss)
        if left!=0:
            cur_loss+=lloss
        if right!=0:
            cur_loss+=rloss
        if upper!=0:
            cur_loss+=uloss
        if lower!=0:
            cur_loss+=dloss
        if args.use_global_color:
            cur_loss+=gloss
        cur_loss *= args.edge_color_weight
        # print(cur_loss,'lsiz')
        # print(cur_loss,'loss1')
        return cur_loss
