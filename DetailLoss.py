
import torch
from torch import nn
import einops
import torchvision
from torch.nn import functional as F

from LossInterface import LossInterface

try:
    from CLIP import clip
except ImportError:
    from clip import clip

# from pixray import parse_prompt, Prompt
from torchvision import transforms


class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply

class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    # print(f"parsed vals is {vals}")
    return vals[0], float(vals[1]), float(vals[2])


class DetailLoss(LossInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    
    @staticmethod
    def add_settings(parser):
        parser.add_argument("-det",  "--detail_prompts", type=str, help="put in prompts to specify detail", default='', dest='detail_prompts')
        return parser
    


    def parse_settings(self,args):
        if args.detail_prompts:
            args.detail_prompts = [phrase.strip() for phrase in args.detail_prompts.split("|")]
        return args

    def add_globals(self,args):
        pmsTable = {}
        perceptors = {}
        
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                      std=[0.26862954, 0.26130258, 0.27577711])

        jit = True if float(torch.__version__[:3]) < 1.8 else False


        for clip_model in args.clip_models:
            perceptor = clip.load(clip_model, jit=jit, download_root="models")[0].eval().requires_grad_(False).to(self.device)
            perceptors[clip_model] = perceptor
                
        for clip_model in args.clip_models:
            pmsTable[clip_model] = []

        for prompt in args.detail_prompts:
            for clip_model in args.clip_models:
                pMs = pmsTable[clip_model]
                perceptor = perceptors[clip_model]
                txt, weight, stop = parse_prompt(prompt)
                embed = perceptor.encode_text(clip.tokenize(txt).to(self.device)).float()
                pMs.append(Prompt(embed, weight, stop).to(self.device))
        
        scale = transforms.Compose([transforms.Scale((224,224))])
        
        lossglobals = {
            'pmsTable': pmsTable,
            'perceptors': perceptors,
            'normalize':    normalize,
            'scale': scale
        }
        return lossglobals


    
    def forward(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        allloss = []
        d_w = 1

        for clip_model in args.clip_models:

            #TODO
            perceptor = lossGlobals['perceptors'][clip_model]

            # cutoutSize = cutoutSizeTable[clip_model]

            # sizedict = [int(i/8) for i in list(out.size())[2:4]]
            sizedict = (51,51)
            # torch.split(tensor_im, (3,30,30))[0].size()
            patches = out.unfold(2, *sizedict).unfold(3, *sizedict) 
            patchtest = patches.reshape([ 3,-1] +list(patches.size()[4:6]))
            allpatches = einops.rearrange(patchtest, 'c b h w -> b c h w').contiguous()
            indices = torch.randperm(len(allpatches))[:20]
            # print(indices)
            
            # print(allpatches.size(),'patches')
            alpat = lossGlobals['scale'](allpatches[indices])
            del allpatches
            # print(allpatches.size(),'patches')


            iii = perceptor.encode_image(lossGlobals['normalize']( alpat )).float()
            del alpat
            pMs = lossGlobals['pmsTable'][clip_model]
            for prompt in pMs:
                allloss.append(prompt(iii)*d_w) 
        return allloss
            
