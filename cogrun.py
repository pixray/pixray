import cog
from pathlib import Path
import torch
import pixray
import yaml
import pathlib
import os

class Predictor(cog.Predictor):
    def setup(self):
        pixray.reset_settings()

    # Define the input types for a prediction
    @cog.input("settings", type=Path, help="Image to classify")
    @cog.input("prompts", type=str, help="New Prompts")
    def predict(self, settings, prompts):
        """Run a single prediction on the model"""
        # settings_file = f"cogs/{settings}.yaml"
        with open(settings, 'r') as stream:
          try:
              base_settings = yaml.safe_load(stream)
          except yaml.YAMLError as exc:
              print("YAML ERROR", exc)
              sys.exit(1)

        pixray.add_settings(**base_settings)
        if prompts is not None and prompts != "":
          pixray.add_settings(prompts=prompts)
        settings = pixray.apply_settings()
        print("---> ", settings.output)
        pixray.do_init(settings)
        pixray.do_run(settings)
        print("---> ", settings.output)
        print(os.getcwd())
        # print(os.path.dirname(os.path.realpath(settings.output)))
        # return pathlib.Path(settings.output)
        return pathlib.Path(os.path.realpath(settings.output))
