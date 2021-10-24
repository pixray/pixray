
import torch
import einops
import torchvision

from LossInterface import LossInterface

try:
    from CLIP import clip
except ImportError:
    from clip import clip

from pixray import parse_prompt, Prompt
from torchvision import transforms


class DetailLoss(LossInterface):
    def __init__(self):
        super().__init__()

    
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

        for prompt in args.prompts:
            for clip_model in args.clip_models:
                pMs = pmsTable[clip_model]
                perceptor = perceptors[clip_model]
                txt, weight, stop = parse_prompt(prompt)
                embed = perceptor.encode_text(clip.tokenize(txt).to(self.device)).float()
                pMs.append(Prompt(embed, weight, stop).to(self.device))
        
        lossglobals = {
            'pmsTable': pmsTable,
            'perceptors': perceptors,
            'normalize':    normalize
        }
        return lossglobals


    
    def forward(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        allloss = []

        for clip_model in args.clip_models:

            #TODO
            perceptor = lossGlobals['perceptors'][clip_model]

            # cutoutSize = cutoutSizeTable[clip_model]

            sizedict = [int(i/3) for i in list(out.size())[2:4]]
            # torch.split(tensor_im, (3,30,30))[0].size()
            patches = out.unfold(2, *sizedict).unfold(3, *sizedict) 
            patchtest = patches.reshape([ 3,-1] +list(patches.size()[4:6]))
            allpatches = einops.rearrange(patchtest, 'c b h w -> b c h w').contiguous()


            iii = perceptor.encode_image(lossGlobals['normalize']( allpatches )).float()

            pMs = lossGlobals['pmsTable'][clip_model]
            for prompt in pMs:
                allloss.append(prompt(iii)) 
        return allloss
            
