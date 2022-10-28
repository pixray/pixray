from cog import Path, BasePredictor, Input
import torch
import pixray
import yaml
import pathlib
import os
import yaml
from typing import Iterator

from cogrun import create_temporary_copy

class GenesisPredictor(BasePredictor):
    def setup(self):
        print("---> GenesisPredictor Setup")

    # Define the input types for a prediction
    def predict(
            self,
            title: str = Input(default=""),
            quality: str =Input(choices=["draft", "mintable"], default="draft"),
            optional_settings: str = Input(default="\n")
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        print("---> Pixray Genesis Init")

        pixray.reset_settings()

        if(quality=="draft"):
            pixray.add_settings(quality="draft", scale=2.5, iterations=100)
        else:
            pixray.add_settings(quality="best", scale=4, iterations=350)

        # apply settings in order
        title = title.strip()
        if title == "" or title == "(untitled)":
            title = "Wow, that looks amazing!|Trending on Artstation"
            pixray.add_settings(custom_loss='saturation')

        # initially assume prompt is title (this can be overridden)
        pixray.add_settings(prompts=title)

        optional_settings = optional_settings.strip()
        if optional_settings != "":
            ydict = yaml.safe_load(optional_settings)
            if ydict is not None:
                print(ydict)
                # add #pixelart to pixel drawer items
                if ("drawer" in ydict) and ydict["drawer"] == "pixel":
                    pixray.add_settings(prompts=f"{title} #pixelart")
                # note settings might explicitly set prompts as well
                pixray.add_settings(**ydict)

        pixray.add_settings(skip_args=True)
        settings = pixray.apply_settings()
        pixray.do_init(settings)
        run_complete = False
        while run_complete == False:
            run_complete = pixray.do_run(settings, return_display=True)
            output_file = os.path.join(settings.outdir, settings.output)
            temp_copy = create_temporary_copy(output_file)
            yield Path(os.path.realpath(temp_copy))
