import re
import argparse
import glob
from pathlib import Path
from braceexpand import braceexpand
from codecs import encode
from PIL import Image
import urllib.request
from colorthief import ColorThief
import subprocess
import datetime
import os

try:
    import matplotlib.colors
except ImportError:
    # only needed for palette stuff
    pass

# https://stackoverflow.com/a/31758803/1010653
class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

# file helpers
def real_glob(rglob):
    glob_list = braceexpand(rglob)
    files = []
    for g in glob_list:
        files = files + glob.glob(g)
    return sorted(files)

def get_file_path(directory, filename, suffix):
    if filename is None or re.search("\\\\|\/|^$", filename.strip()):
        raise ValueError("Invalid filename specified.")

    return str(Path(directory, filename).with_suffix(suffix))

####### argparse bools ##########
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


####### PALETTE SECTION ##########

# canonical interpolation function, like https://p5js.org/reference/#/p5/map
def map_number(n, start1, stop1, start2, stop2):
  return ((n-start1)/(stop1-start1))*(stop2-start2)+start2;

# parses either (255,255,0) or [1,1,0] as yellow, etc
def parse_triple_to_rgb(s):
    s2 = re.sub(r'[(\[\])]', '', s)
    t = s2.split("+")
    rgb = [float(n) for n in t]
    if s[0] == "(":
        rgb = [n / 255.0 for n in rgb]
    return rgb

# here are examples of what can be parsed
# white   (16 color black to white ramp)
# red     (16 color black to red ramp)
# rust\8  (8 color black to rust ramp)
# red->rust         (16 color red to rust ramp)
# red->#ff0000      (16 color red to yellow ramp)
# red->#ff0000\20   (20 color red to yellow ramp)
# black->red->white (16 color black/red/white ramp)
# [black, red, #ff0000] (three colors)
# red->white;blue->yellow (32 colors across two ramps of 16)
# red;blue;yellow         (48 colors from combining 3 ramps)
# red\8;blue->yellow\8    (16 colors from combining 2 ramps)
# red->yellow;[black]     (16 colors from ramp and also black)
#
# TODO: maybe foo.jpg, foo.json, foo.png, foo.asc
def get_single_rgb(s):
    palette_lookups = {
        "pixel_green":     [0.44, 1.00, 0.53],
        "pixel_orange":    [1.00, 0.80, 0.20],
        "pixel_blue":      [0.44, 0.53, 1.00],
        "pixel_red":       [1.00, 0.53, 0.44],
        "pixel_grayscale": [1.00, 1.00, 1.00],
    }
    if s[0] == "("  or s[0] == "[":
        rgb = parse_triple_to_rgb(s)
    elif s in palette_lookups:
        rgb = palette_lookups[s]
    elif s[:4] == "mat:":
        rgb = matplotlib.colors.to_rgb(s[4:])
    elif matplotlib.colors.is_color_like(f"xkcd:{s}"):
        rgb = matplotlib.colors.to_rgb(f"xkcd:{s}")
    else:
        rgb = matplotlib.colors.to_rgb(s)
    return rgb

def expand_colors(colors, num_steps):
    index_episilon = 1e-6;
    pal = []
    num_colors = len(colors)
    for n in range(num_steps):
        cur_float_index = map_number(n, 0, num_steps-1, 0, num_colors-1)
        cur_int_index = int(cur_float_index)
        cur_float_offset = cur_float_index - cur_int_index
        if(cur_float_offset < index_episilon or (1.0-cur_float_offset) < index_episilon):
            # debug print(n, "->", cur_int_index)
            pal.append(colors[cur_int_index])
        else:
            # debug print(n, num_steps, num_colors, cur_float_index, cur_int_index, cur_float_offset)
            rgb1 = colors[cur_int_index]
            rgb2 = colors[cur_int_index+1]
            r = map_number(cur_float_offset, 0, 1, rgb1[0], rgb2[0])
            g = map_number(cur_float_offset, 0, 1, rgb1[1], rgb2[1])
            b = map_number(cur_float_offset, 0, 1, rgb1[2], rgb2[2])
            pal.append([r, g, b])
    return pal

def get_rgb_range(s):
    # get the list that defines the range
    if s.find('->') > 0:
        parts = s.split('->')
    else:
        parts = ["black", s]

    # look for a number of parts at the end
    if parts[-1].find('\\') > 0:
        colname, steps = parts[-1].split('\\')
        parts[-1] = colname
        num_steps = int(steps)
    else:
        num_steps = 16

    colors = [get_single_rgb(s) for s in parts]
    #debug print("We have colors: ", colors)

    pal = expand_colors(colors, num_steps)
    return pal

def palette_from_section(s):
    s = s.strip()
    if s[0] == '[':
        # look for a number of parts at the end
        if s.find('\\') > 0:
            col_list, steps = s.split('\\')
            s = col_list
            num_steps = int(steps)
        else:
            num_steps = None

        chunks = s[1:-1].split(",")
        # chunks = [s.strip().tolower() for c in chunks]
        pal = [get_single_rgb(c.strip()) for c in chunks]

        if num_steps is not None:
            pal = expand_colors(pal, num_steps)

        return pal
    elif s[0] == '@' or s[0:4] == 'http':

        # look for a number of parts at the end
        if s.find('\\') > 0:
            stem, steps = s.split('\\')
            s = stem
            num_steps = int(steps)
        else:
            num_steps = None

        # pull from a file or URL
        if s.endswith(".png") or s.endswith(".jpg"):
            if 'http' in s:
                opener = AppURLopener()
                obj = opener.open(s)
            else:
                obj = s[1:]

            if num_steps is None:
                num_steps = 255

            # https://stackoverflow.com/a/22442212/1010653
            # try to get all the colors
            img = Image.open(obj)
            img = img.convert('RGB')
            color_pairs = img.getcolors(img.size[0]*img.size[1])
            print(f"Found {len(color_pairs)} colors in {s}")
            if len(color_pairs) <= num_steps:
                # not too many
                colors = [c[1] for c in color_pairs]
                colors = [[c[0]/255.0, c[1]/255.0, c[2]/255.0] for c in colors]
                return colors

            # dumb: but colortheif can't deal with an Image, so reopen the url if need be
            if 'http' in s:
                opener = AppURLopener()
                obj = opener.open(s)
            ct = ColorThief(obj)
            colors = ct.get_palette(color_count=num_steps, quality=2)
            colors = [[c[0]/255.0, c[1]/255.0, c[2]/255.0] for c in colors]
            print(f"Distilled {len(colors)} (reqeusted {num_steps}) colors from {s}")
            return list(colors)

        elif s.endswith(".act"):
            # https://stackoverflow.com/a/48873783/1010653
            if s[0] == '@':
                with open(s[1:], 'rb') as act:
                    raw_data = act.read()                           # Read binary data
            else:
                opener = AppURLopener()
                obj = opener.open(s)
                raw_data = obj.read()
            hex_data = encode(raw_data, 'hex')                  # Convert it to hexadecimal values
            total_colors_count = (int(hex_data[-7:-4], 16))     # Get last 3 digits to get number of colors total
            misterious_count = (int(hex_data[-4:-3], 16))       # I have no idea what does it do
            colors_count = (int(hex_data[-3:], 16))             # Get last 3 digits to get number of nontransparent colors

            # Decode colors from hex to string and split it by 6 (because colors are #1c1c1c)
            colors = [hex_data[i:i+6].decode() for i in range(0, total_colors_count*6, 6)]

            # Add # to each item and filter empty items if there is a corrupted total_colors_count bit
            colors = ['#'+i for i in colors if len(i)]
            colors = [get_single_rgb(s) for s in colors]
            return colors
        else:
            raise ValueError(f'Unknown file type: {s}')
    else:
        return get_rgb_range(s)
 
def palette_from_string(s):
    s = s.strip()
    pal = []
    chunks = s.split(';')
    for c in chunks:
        pal = pal + palette_from_section(c)
    return pal


# downloader
def wget_file(url, out):
    try:
        print(f"Downloading {out} from {url}, please wait")
        output = subprocess.check_output(['wget', '-O', out, url])
    except subprocess.CalledProcessError as cpe:
        output = cpe.output
        print("Ignoring non-zero exit: ", output)

# filename templates - can fill in placeholders for %DATE%, %SIZE% and %SEQ%
# TODO: load more of these settings into the template_dict
def emit_filename(filename, template_dict={}, args=None):
    datestr = datetime.datetime.now().strftime("%Y%m%d")
    filename = filename.replace('%DATE%', datestr)

    for key in template_dict:
        pattern = "%{}%".format(key)
        value = "{}".format(template_dict[key])
        filename = filename.replace(pattern, value)

    if args is not None:
        # legacy version - needs updating!
        if args.model:
            model = args.model.replace(".", "_")
        else:
            model = "NoModel"
        if args.seed:
            seed = "{:d}".format(args.seed)
        else:
            seed = "NoSeed"
        if '%MODEL%' in filename:
            filename = filename.replace('%MODEL%', model)
        if '%OFFSET%' in filename:
            filename = filename.replace('%OFFSET%', "{:d}".format(args.offset))
        if '%SEED%' in filename:
            filename = filename.replace('%SEED%', seed)
        if '%ROWS%' in filename:
            filename = filename.replace('%ROWS%', "{:d}".format(args.rows))
        if '%COLS%' in filename:
            filename = filename.replace('%COLS%', "{:d}".format(args.cols))
        if '%INDEX%' in filename:
            filename = filename.replace('%INDEX%', "{}".format(args.anchor_offset_x))
    if '%SEQ%' in filename:
        # determine what the next available number is
        cur_seq = 1
        candidate = filename.replace('%SEQ%', "{:02d}".format(cur_seq))
        while os.path.exists(candidate):
            cur_seq = cur_seq + 1
            candidate = filename.replace('%SEQ%', "{:02d}".format(cur_seq))
        filename = candidate
    return filename
