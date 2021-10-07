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
        os.environ['TRANSFORMERS_CACHE'] = 'models/'
        pixray.reset_settings()

    # Define the input types for a prediction
    @cog.input("settings", type=str, help="Image to classify")
    @cog.input("prompts", type=str, help="New Prompts")
    def predict(self, settings, prompts):
        """Run a single prediction on the model"""
        print("---> BasePixrayPredictor Predict")
        settings_file = f"cogs/{settings}.yaml"
        with open(settings_file, 'r') as stream:
          try:
              base_settings = yaml.safe_load(stream)
          except yaml.YAMLError as exc:
              print("YAML ERROR", exc)
              sys.exit(1)

        pixray.add_settings(**base_settings)
        if prompts is not None and prompts != "":
          pixray.add_settings(prompts=prompts)
        settings = pixray.apply_settings()
        pixray.do_init(settings)
        pixray.do_run(settings)
        temp_copy = create_temporary_copy(settings.output)
        print(temp_copy)
        return pathlib.Path(os.path.realpath(temp_copy))

class PixrayVqgan(BasePixrayPredictor):
    @cog.input("prompts", type=str, help="New Prompts")
    def predict(self, prompts):
        super().predict(settings="pixray_vqgan", prompts=prompts)