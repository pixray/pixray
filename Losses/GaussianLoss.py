import torch
from Losses.LossInterface import LossInterface


def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern(ylen=256, xlen=256, stdy=128, stdx=128):
    """Returns a 2D Gaussian kernel array."""
    gkerny = gaussian_fn(ylen, std=stdy) 
    gkernx = gaussian_fn(xlen, std=stdx) 
    gkern2d = torch.outer(gkerny, gkernx)
    return gkern2d

class GaussianLoss(LossInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("--gaussian_weight", type=float, help="gaussian's weight", default=1, dest='gaussian_weight')
        parser.add_argument("--gaussian_std", nargs=2 ,type=float, help="gaussian's std for both x and y", default=(40,40), dest='gaussian_std')
        parser.add_argument("--gaussian_color", nargs=3 ,type=float, help="color for gaussian to optimize to", default=(255,255,255), dest='gaussian_color')
        return parser
    

    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        gaus = gkern(out.size()[2],out.size()[3], *args.gaussian_std).to(self.device)
        color = torch.zeros(out.size()).to(self.device)
        Rval,Gval,Bval = args.gaussian_color
        color[:,0,:,:] = Rval/255
        color[:,1,:,:] = Gval/255
        color[:,2,:,:] = Bval/255
        # mseloss = nn.MSELoss()
        loss = torch.abs(out - color)
        # print(loss.size()) 
        loss = loss*torch.abs(1-gaus)

        cur_loss = torch.mean(loss)
        return cur_loss*args.gaussian_weight
