import cog
from pathlib import Path
import torch
import pixray
import yaml
import pathlib
import os

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
            temp_copy = create_temporary_copy(settings.output)
            yield pathlib.Path(os.path.realpath(temp_copy))

class PixrayVqgan(BasePixrayPredictor):
    @cog.input("prompts", type=str, help="text prompt", default="rainbow mountain")
    @cog.input("quality", type=str, help="better is slower", default="normal", options=["draft", "normal", "better", "best"])
    @cog.input("aspect", type=str, help="Wide vs square", default="widescreen", options=["widescreen", "square"])
    def predict(self, **kwargs):
        yield from super().predict(settings="pixray_vqgan", **kwargs)

class PixrayPixel(BasePixrayPredictor):
    @cog.input("prompts", type=str, help="text prompt", default="Beirut Skyline. #pixelart")
    @cog.input("aspect", type=str, help="wide vs square", default="widescreen", options=["widescreen", "square"])
    @cog.input("drawer", type=str, help="render engine", default="pixel", options=["pixel", "vqgan", "line_sketch", "clipdraw"])
    def predict(self, **kwargs):
        yield from super().predict(settings="pixray_pixel", **kwargs)