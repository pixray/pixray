# pixray

-![Alt text](https://user-images.githubusercontent.com/945979/132935558-1e03178a-7e45-4dde-8e5d-f9b7b9d60d74.png "pixray banner")

Pixray is an image generation system. It combines previous ideas including:

 * [Perception Engines](https://github.com/dribnet/perceptionengines) on iteratively optimising graphics against on an ensemble of classifiers
 * [CLIP guided GAN imagery](https://alexasteinbruck.medium.com/vqgan-clip-how-does-it-work-210a5dca5e52) from [Ryan Murdoch](https://twitter.com/advadnoun) and [Katherine Crowson](https://github.com/crowsonkb) as well as modifictions such as [CLIPDraw](https://twitter.com/kvfrans/status/1409933704856674304) from Kevin Frans
 * [Sampling Generative Networks](https://github.com/dribnet/plat) on useful ways of navigating latent space
 * (more to come)

pixray it itself a python library and command line utility, but is also friendly to running on line in Google Colab notebooks.

The system is currently lacking documentation. Instead plese checkout [THE DEMO NOTEBOOKS](https://github.com/dribnet/clipit/tree/master/demos) - especially the super simple "Start Here" colab.


# Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```
```bibtex
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis}, 
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

