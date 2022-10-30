# pixray

![Alt text](https://user-images.githubusercontent.com/945979/132954388-1986e4c6-6996-48fd-9e91-91ec97963781.png "deep ocean monsters #pixelart")

Pixray is an image generation system. It combines previous ideas including:

 * [Perception Engines](https://github.com/dribnet/perceptionengines) which uses image augmentation and iteratively optimises images against an ensemble of classifiers
 * [CLIP guided GAN imagery](https://alexasteinbruck.medium.com/vqgan-clip-how-does-it-work-210a5dca5e52) from [Ryan Murdoch](https://twitter.com/advadnoun) and [Katherine Crowson](https://github.com/crowsonkb) as well as modifictions such as [CLIPDraw](https://twitter.com/kvfrans/status/1409933704856674304) from Kevin Frans
 * Useful ways of navigating latent space from [Sampling Generative Networks](https://github.com/dribnet/plat)
 * (more to come)

pixray is itself a python library and command line utility, but is also friendly to running online in Google Colab notebooks.

There is currently [some documentation on options](https://dazhizhong.gitbook.io/pixray-docs/docs). Also checkout [THE DEMO NOTEBOOKS](https://github.com/pixray/pixray_notebooks) or join in the [discussion on discord](https://discord.gg/x2g9TWrNKe).

## Usage

Be sure to `git clone --recursive` to also get submodules.

You can install `pip install -r requirements.txt` and then `pip install basicsr` manually in a fresh python 3.8 environment (eg: using conda). After that you can use the included `pixray.py` command line utility:

    python pixray.py --drawer=pixel --prompt=sunrise --outdir sunrise01

pixray can also be run from within your own python code, like this

```python
import pixray
pixray.run("an extremely hairy panda bear", "vdiff", custom_loss="aesthetic", outdir="outputs/hairout")
```

Examples of pixray colab notebooks can be found [in this separate repo](https://github.com/pixray/pixray_notebooks).

running in a Docker using [Cog](https://github.com/replicate/cog) is also possible. First, [install Docker and Cog](https://github.com/replicate/cog#install), then you can use `cog run` to run Pixray inside Docker. For example: 

    cog run python pixray.py --drawer=pixel --prompt=sunrise --outdir sunrise01
