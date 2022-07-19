import torch

from Losses.LossInterface import LossInterface


class SaturationLoss(LossInterface):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def add_settings(parser):
        parser.add_argument(
            "--saturation_weight",
            type=float,
            help="strength of pallete loss effect",
            default=1,
            dest="saturation_weight",
        )
        return parser

    def get_loss(self, cur_cutouts, out, args, globals=None, lossGlobals=None):
        # based on the old "percepted colourfulness" heuristic from Hasler and Süsstrunk’s 2003 paper
        # https://www.researchgate.net/publication/243135534_Measuring_Colourfulness_in_Natural_Images
        all_loss = []
        for _, cutouts in cur_cutouts.items():
            _pixels = cutouts.permute(0, 2, 3, 1).reshape(-1, 3)
            rg = _pixels[:, 0] - _pixels[:, 1]
            yb = 0.5 * (_pixels[:, 0] + _pixels[:, 1]) - _pixels[:, 2]
            rg_std, rg_mean = torch.std_mean(rg)
            yb_std, yb_mean = torch.std_mean(yb)
            std_rggb = torch.sqrt(rg_std**2 + yb_std**2)
            mean_rggb = torch.sqrt(rg_mean**2 + yb_mean**2)
            colorfullness = std_rggb + 0.3 * mean_rggb
            all_loss.append(-colorfullness * args.saturation_weight / 10.0)

        return all_loss
