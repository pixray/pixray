import cog
from pathlib import Path
import torch
import yaml
import pathlib
import os
import yaml

# https://stackoverflow.com/a/6587648/1010653
import tempfile, shutil
def create_temporary_copy(src_path):
    _, tf_suffix = os.path.splitext(src_path)
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"tempfile{tf_suffix}")
    shutil.copy2(src_path, temp_path)
    return temp_path

class BasePixrayPredictor(cog.Predictor):
    def setup(self):
        print("---> BasePixrayPredictor Setup")
        os.environ['TORCH_HOME'] = 'models/'

    # Define the input types for a prediction
    @cog.input("settings", type=str, help="Default settings to use")
    @cog.input("prompts", type=str, help="Text Prompts")
    def predict(self, settings, **kwargs):
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
            yield pathlib.Path(os.path.realpath(temp_copy))

class PixrayVqgan(BasePixrayPredictor):
    @cog.input("prompts", type=str, help="text prompt", default="rainbow mountain")
    @cog.input("quality", type=str, help="better is slower", default="normal", options=["draft", "normal", "better", "best"])
    @cog.input("aspect", type=str, help="wide vs square", default="widescreen", options=["widescreen", "square"])
    # @cog.input("num_cuts", type=int, default="24", min=4, max=96)
    # @cog.input("batches", type=int, default="1", min=1, max=32)
    def predict(self, **kwargs):
        yield from super().predict(settings="pixray_vqgan", **kwargs)

class PixrayPixel(BasePixrayPredictor):
    @cog.input("prompts", type=str, help="text prompt", default="Beirut Skyline. #pixelart")
    @cog.input("aspect", type=str, help="wide vs square", default="widescreen", options=["widescreen", "square"])
    @cog.input("drawer", type=str, help="render engine", default="pixel", options=["pixel", "vqgan", "line_sketch", "clipdraw"])
    def predict(self, **kwargs):
        yield from super().predict(settings="pixray_pixel", **kwargs)

class Text2Image(BasePixrayPredictor):
    @cog.input("prompts", type=str, help="description of what to draw", default="Robots skydiving high above the city")
    @cog.input("quality", type=str, help="speed vs quality", default="normal", options=["draft", "normal", "better", "best"])
    @cog.input("aspect", type=str, help="wide or narrow", default="widescreen", options=["widescreen", "square", "portrait"])
    def predict(self, **kwargs):
        yield from super().predict(settings="text2image", **kwargs)

class Text2Pixel(BasePixrayPredictor):
    @cog.input("prompts", type=str, help="text prompt", default="Manhattan skyline at sunset. #pixelart")
    @cog.input("aspect", type=str, help="wide or narrow", default="widescreen", options=["widescreen", "square", "portrait"])
    @cog.input("pixel_scale", type=float, help="bigger pixels", default=1.0, min=0.5, max=2.0)
    def predict(self, **kwargs):
        yield from super().predict(settings="text2pixel", **kwargs)

class PixrayRaw(BasePixrayPredictor):
    @cog.input("prompts", type=str, help="text prompt", default="Manhattan skyline at sunset. #pixelart")
    @cog.input("settings", type=str, help="yaml settings", default='\n')
    def predict(self, prompts, settings):
        ydict = yaml.safe_load(settings)
        if ydict == None:
            # no settings
            ydict = {}
        yield from super().predict(settings="pixrayraw", prompts=prompts, **ydict)

class PixrayApi(BasePixrayPredictor):
    @cog.input("settings", type=str, help="yaml settings", default='\n')
    def predict(self, settings):
        ydict = yaml.safe_load(settings)
        if ydict == None:
            # no settings
            ydict = {}
        yield from super().predict(settings="pixrayapi", **ydict)

class Tiler(BasePixrayPredictor):
    @cog.input("prompts", type=str, help="text prompt", default="Beautiful marble texture")
    @cog.input("pixelart", type=bool, help="pixelart style?", default=False)
    @cog.input("mirror", type=bool, help="shifted pattern?", default=False)
    @cog.input("settings", type=str, help="yaml settings", default='\n')
    def predict(self, prompts, pixelart, mirror, settings):
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
    @cog.input("prompts", type=str, help="text prompt", default="Manhattan skyline at sunset. #artstation 🌇")
    @cog.input("settings", type=str, help="extra settings in `name: value` format. reference: https://dazhizhong.gitbook.io/pixray-docs/docs/primary-settings", default='\n')
    def predict(self, prompts, settings):
        ydict = yaml.safe_load(settings)
        if ydict == None:
            # no settings
            ydict = {}
        yield from super().predict(settings="pixray_vdiff", prompts=prompts, **ydict)

