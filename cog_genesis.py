import cog
from pathlib import Path
import torch
import pixray
import yaml
import pathlib
import os
import yaml

from cogrun import create_temporary_copy

class GenesisPredictor(cog.Predictor):
    def setup(self):
        print("---> GenesisPredictor Setup")

    # Define the input types for a prediction
    @cog.input("prompt", type=str, help="say anything", default="")
    @cog.input("style", type=str, options=["image", "pixels"], default="image")
    @cog.input("version", type=str, options=["practice", "final"], default="practice")
    @cog.input("settings", type=str, help="(optional)", default="\n")
    def predict(self, prompt, style, version, settings):
        """Run a single prediction on the model"""
        print("---> GenesisPredictor Predict")

        pixray.reset_settings()

        if(version=="practice"):
            pixray.add_settings(output="outputs/genesis_draft.png", quality="draft", scale=2.5, iterations=100)
        else:
            pixray.add_settings(output="outputs/genesis.png", quality="best", scale=4, iterations=300)

        empty_settings = True
        # apply settings in order
        prompt = prompt.strip()
        if prompt != "":
            pixray.add_settings(prompts=prompt)
            empty_settings = False

        if style == "image":
            pixray.add_settings(drawer="vqgan")
        else:
            pixray.add_settings(drawer="pixel")

        settings = settings.strip()
        if settings != "":
            ydict = yaml.safe_load(settings)
            if ydict is not None:
                pixray.add_settings(**ydict)
                empty_settings = False

        # TODO: something if empty_settings?

        if empty_settings == True:
            pixray.add_settings(prompts="Wow, that looks amazing!|Trending on Artstation")
            pixray.add_settings(custom_loss='saturation')

        pixray.add_settings(skip_args=True)
        settings = pixray.apply_settings()
        pixray.do_init(settings)
        run_complete = False
        while run_complete == False:
            run_complete = pixray.do_run(settings, return_display=True)
            temp_copy = create_temporary_copy(settings.output)
            yield pathlib.Path(os.path.realpath(temp_copy))
