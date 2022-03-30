from cog import BasePredictor, Input, Path
from typing import Iterator
import torch
import yaml
import pathlib
import os
import yaml

from util import get_single_rgb

# https://stackoverflow.com/a/6587648/1010653
import tempfile, shutil
def create_temporary_copy(src_path):
    _, tf_suffix = os.path.splitext(src_path)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"tempfile{tf_suffix}")
    shutil.copy2(src_path, temp_path)
    return temp_path

class BasePixrayPredictor(BasePredictor):
    def setup(self):
        print("---> BasePixrayPredictor Setup")
        os.environ['TORCH_HOME'] = 'models/'

    def predict(self, 
        settings: str = Input(description="Default settings to use"),
    **kwargs) -> Iterator[Path]:
        # workaround for import issue when deploying
        import pixray
        """Run a single prediction on the model"""
        print("---> BasePixrayPredictor Predict")
        os.environ['TORCH_HOME'] = 'models/'
        settings_file = f"cogs/{settings}.yaml"
        with open(settings_file, 'r') as stream:
          try:
              base_settings = yaml.safe_load(stream)
          except yaml.YAMLError as exc:
              print("YAML ERROR", exc)
              sys.exit(1)

        pixray.reset_settings()
        pixray.add_settings(**base_settings)
        pixray.add_settings(**kwargs)
        pixray.add_settings(skip_args=True)
        settings = pixray.apply_settings()
        pixray.do_init(settings)
        run_complete = False
        while run_complete == False:
            run_complete = pixray.do_run(settings, return_display=True)
            output_file = os.path.join(settings.outdir, settings.output)
            temp_copy = create_temporary_copy(output_file)
            yield Path(os.path.realpath(temp_copy))

class PixrayVqgan(BasePixrayPredictor):
    def predict(self, 
        prompts: str = Input(description="text prompt", default="rainbow mountain"),
        quality: str = Input(description="better is slower", default="normal", choices=["draft", "normal", "better", "best"]),
        aspect: str = Input(description="wide vs square", default="widescreen", choices=["widescreen", "square"]),
        # num_cuts: int = Input(description="number of cuts", default=24, ge:4, le:96),
        # batches: int = Input(description="number of batches", default=1, ge:1, le:32),
        **kwargs
    ) -> Iterator[Path]:
        yield from super().predict(settings="pixray_vqgan", **kwargs)

class PixrayPixel(BasePixrayPredictor):
    def predict(self, 
        prompts: str = Input(description="text promps", default="Beirut Skyline. #pixelart"),
        aspect: str = Input(description="wide vs square", default="widescreen", choices=["widescreen", "square"]),
        drawer: str = Input(description="render engine", default="pixel", choices=["pixel", "vqgan", "line_sketch", "clipdraw"]),
        **kwargs
    ) -> Iterator[Path]:
        yield from super().predict(settings="pixray_pixel", **kwargs)

class Text2Image(BasePixrayPredictor):
    def predict(self, 
        prompts: str = Input(description="text prompt", default="Cairo skyline at sunset."),
        drawer: str = Input(description="render engine", default="vqgan", choices=["pixel", "vqgan", "vdiff", "fft", "fast_pixel", "line_sketch", "clipdraw"]),
        settings: str = Input(description="extra settings in `name: value` format. reference: https://dazhizhong.gitbook.io/pixray-docs/docs/primary-settings", default='\n')
    ) -> Iterator[Path]:
        ydict = yaml.safe_load(settings)
        if ydict == None:
            # no settings
            ydict = {}
        yield from super().predict(settings="text2image", prompts=prompts, drawer=drawer, **ydict)

class Text2Pixel(BasePixrayPredictor):
    def predict(self, 
        prompts: str = Input(description="text prompt", default="Manhattan skyline at sunset. #pixelart"),
        aspect: str = Input(description="wide or narrow", default="widescreen", choices=["widescreen", "square", "portrait"]),
        pixel_scale: float = Input(description="bigger pixels", default=1.0, ge=0.5, le=2.0),
        **kwargs
    ) -> Iterator[Path]:
        yield from super().predict(settings="text2pixel", **kwargs)

class PixrayRaw(BasePixrayPredictor):
    def predict(self, 
        prompts: str = Input(description="text prompt", default="Manhattan skyline at sunset. #pixelart"),
        settings: str = Input(description="yaml settings", default="\n")
    ) -> Iterator[Path]:
        ydict = yaml.safe_load(settings)
        if ydict == None:
            # no settings
            ydict = {}
        yield from super().predict(settings="pixrayraw", prompts=prompts, **ydict)

class PixrayApi(BasePixrayPredictor):
    def predict(self, 
        settings: str = Input(description="yaml settings", default="\n")
    ) -> Iterator[Path]:
        ydict = yaml.safe_load(settings)
        if ydict == None:
            # no settings
            ydict = {}
        yield from super().predict(settings="pixrayapi", **ydict)

class Tiler(BasePixrayPredictor):
    def predict(self, 
        prompts: str = Input(description="text prompt", default=""),
        pixelart: bool = Input(description="pixelart style?", default=False),
        mirror: bool = Input(description="shifted pattern?", default=False),
        settings: str = Input(description="yaml settings", default="\n")
    ) -> Iterator[Path]:
        ydict = yaml.safe_load(settings)
        if ydict == None:
            # no settings
            ydict = {}
        if pixelart:
            if mirror:
                settings = "tiler_pixel_shift"
            else:
                settings = "tiler_pixel"
            yield from super().predict(prompts=f"{prompts} #pixelart", settings=settings, **ydict)
        else:
            if mirror:
                settings = "tiler_fft_shift"
            else:
                settings = "tiler_fft"
            yield from super().predict(prompts=prompts, settings=settings, **ydict)    

class PixrayVdiff(BasePixrayPredictor):
    def predict(self, 
        prompts: str = Input(description="text prompt", default="Manhattan skyline at sunset. #artstation 🌇"),
        settings: str = Input(description="extra settings in `name: value` format. reference: https://dazhizhong.gitbook.io/pixray-docs/docs/primary-settings", default='\n')
    ) -> Iterator[Path]:
        ydict = yaml.safe_load(settings)
        if ydict == None:
            # no settings
            ydict = {}
        yield from super().predict(settings="pixray_vdiff", prompts=prompts, **ydict)

class EightBidG(BasePixrayPredictor):
    def predict(self, 
        prompts: str = Input(description="text prompt", default=""),
        palette: str = Input(description="colors to use", default="full color", choices=["full color", "web safe", "grayscale"]),
        border: str = Input(description="border color", default="none", choices=["white", "black", "grey", "none"]),
    ) -> Iterator[Path]:
        ydict = {}
        if border == "none":
            ydict.update({"custom_loss":"smoothness:0.25"})
        else:
            ydict.update({"custom_loss":"edge,smoothness:0.25", "edge_thickness":2, "edge_color":get_single_rgb(border)})
        if palette == "grayscale":
            ydict.update({"filters":"lookup", "palette":'black->white\256'})
        elif palette == "web safe":
            ydict.update({"filters":"lookup", "palette":'https://www.pagetutor.com/common/net216pics/net216.gif'})
        yield from super().predict(prompts=prompts, settings="8bidg", **ydict)
