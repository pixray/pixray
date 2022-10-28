#!/bin/bash
set -eux -o pipefail

# Run a prediction to make sure the model works, then push it to Replicate
function deploy {
    class=$1
    shift
    name=$1
    shift
    test_args=$@

    cat cog.yaml.in | sed "s;__CLASS__;$class;" | sed "s;__NAME__;$name;" > cog.yaml
    cog predict $test_args

    if [ ! -f output.1.png ]; then
        echo "No output produced, exiting"
        exit 1
    fi

    cog push

    rm output.*.png
}

deploy cogrun.py:EightBidG dribnet/8bidoug -i prompts="test" -i palette="grayscale"
deploy cogrun.py:PixrayApi dribnet/pixray-api -i prompts="test"
deploy cogrun.py:PixrayRaw dribnet/pixray -i prompts="test"
deploy cogrun.py:Tiler dribnet/pixray-tiler -i prompts="test" -i pixelart=true
deploy cog_genesis.py:GenesisPredictor dribnet/pixray-genesis -i title="test" -i quality="draft"
deploy cogrun.py:Text2Pixel dribnet/pixray-text2pixel -i prompts="test" -i pixel_scale="1.5"
deploy cogrun.py:Text2Image dribnet/pixray-text2image -i prompts="test" -i drawer="clipdraw"
deploy cogrun.py:PixrayPixel dribnet/pixray-pixel -i prompts="test" -i drawer="pixel"
deploy cogrun.py:PixrayVqgan dribnet/pixray-vqgan -i prompts="test" -i quality=draft -i aspect=square
