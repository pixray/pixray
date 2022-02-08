import os
import torch
import torch.nn.functional as F
from Losses.LossInterface import LossInterface
from torchvision import transforms
from util import wget_file
from blip.blip_itm import blip_itm


blip_checkpoint_table = {
    "model_base_retrieval_coco": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth",
}


def parse_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])


class BLIPLoss(LossInterface):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.blip_model = "model_base_retrieval_coco"

        checkpoint_path = f'models/blip_{self.blip_model}.ckpt'
        if not os.path.exists(checkpoint_path):
            wget_file(blip_checkpoint_table[self.blip_model], checkpoint_path)

        self.image_size = 384
        self.model = blip_itm(pretrained=checkpoint_path, image_size=self.image_size, vit='base')
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)

        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    @staticmethod
    def add_settings(parser):
        return parser

    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        text_prompts, weights, _ = zip(*[parse_prompt(prompt) for prompt in args.prompts])

        text = self.model.tokenizer(
            text_prompts,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )

        text_output_itc = self.model.text_encoder(
            text.input_ids.to(self.device),
            attention_mask=text.attention_mask.to(self.device),
            return_dict=True,
            mode="text",
        )

        text_features = F.normalize(
            self.model.text_proj(text_output_itc.last_hidden_state[:, 0, :]),
            dim=-1,
        )

        max_size = max([size for size in cur_cutouts.keys()])
        images = cur_cutouts[max_size]
        if images.shape[-2:] != (self.image_size, self.image_size):
            images = F.interpolate(
                images,
                size=self.image_size,
                mode="bicubic",
                align_corners=False,
            )

        image_embeds = self.model.visual_encoder(self.normalize(images))

        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            self.device
        )

        text_output_itm = self.model.text_encoder(
            text.input_ids.to(self.device).repeat(len(images), 1),
            attention_mask=text.attention_mask.repeat(len(images), 1).to(
                self.device
            ),
            encoder_hidden_states=image_embeds.repeat(len(text_prompts), 1, 1),
            encoder_attention_mask=image_atts.repeat(len(text_prompts), 1),
            return_dict=True,
        )
        itm_loss = -(F.softmax(  # softmax in original. optimizing logit gives it a huge strength
            self.model.itm_head(
                text_output_itm.last_hidden_state[:, 0, :].to(self.device)
            ),
            dim=1,
        )[:, 1] * torch.tensor(weights).repeat(len(images)).to(self.device)).mean()

        image_features = F.normalize(
            self.model.vision_proj(image_embeds[:, 0, :]), dim=-1
        )

        spherical_distance_itc = (
            (image_features[None, :] - text_features[:, None])
            .norm(dim=-1)
            .div(2)
            .arcsin()
            .square()
            .mul(2)
            .mul(torch.tensor(weights)[:, None].to(self.device))
        ).mean()
        return (spherical_distance_itc + itm_loss) / 2
