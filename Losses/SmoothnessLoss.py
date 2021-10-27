import torch
from LossInterface import LossInterface

class SmoothnessLoss(LossInterface):
    def __init__(self,custom_init,**kwargs):
        super().__init__(**kwargs)
    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("-smo",  "--smoothness", type=float, help="encourage smoothness, 0 -- skip", default=0, dest='smoothness')
        parser.add_argument("-est",  "--smoothness_type", type=str, help="enforce smoothness type: default/clipped/log", default='default', dest='smoothness_type')
        return parser

    
    def forward(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        cur_loss = []
        for _,cutouts in cur_cutouts.items():
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
            cur_loss.append(sharpness*args.smoothness)

        return cur_loss
