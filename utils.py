import os
import cv2
import numpy as np
from align import AlignType
from tonemapping import NormalizeType

def to_bool(value:str, line):
    try:
        return bool(value)
    except ValueError:
        print(f"Error: {value} is not a valid bool value, Line {line+1}")
        exit()

def to_int(value:str, line):
    try:
        return int(value)
    except ValueError:
        print(f"Error: {value} is not a valid int value, Line {line+1}")
        exit()

def to_float(value:str, line):
    try:
        return float(value)
    except ValueError:
        print(f"Error: {value} is not a valid float value, Line {line+1}")
        exit()

def read_ldr_images(source_dir:str) -> tuple[list[cv2.Mat | np.ndarray[np.uint8, 3]], int, np.ndarray[np.float32], AlignType, int, int]:
    """
    Read the image_list.txt and read all images included in the list. Then converts images into r,g,b channels and log of exposure times

    Parameters:
    source_dir : the path of directory containing image_list.txt and LDR images

    Returns:
    images[j] : the jth image
    lnt[j]    : the log delta t or log shutter speed for image j
    l         : the lambda (smoothness) used in solving response curve
    alignType : the alignment method to use
    std_ing_idx : the standard image's index in the list of images (for alignment)
    depth     : the recursion depth of MTB algorithm
    """

    filepaths = []
    exposure_times = []
    l = 20
    alignType = AlignType.NONE
    std_img_idx = -1
    depth = 3

    try:
        with open(os.path.join(source_dir, 'image_list.txt'), 'r') as img_list:
            for i, line in enumerate(img_list):
                line = line.split('#')[0].strip() # remove the comments starts with '#'
                if len(line) == 0:
                    continue
                if line.startswith('ALIGN'):
                    line.replace(' ', '')
                    t = line.split('=')[1]
                    if t == '1' or t == 'OUR':
                        alignType = AlignType.OUR
                    elif t == '2' or t == 'CV2':
                        alignType = AlignType.CV2
                    elif t != '0' and t != 'NONE':
                        print(f"Error: AlignType {t} not found in image_list.txt, Line {i+1}")
                        exit()
                elif line.startswith('STD'):
                    line.replace(' ', '')
                    std_img_idx = to_int(line.split('=')[1], i)
                elif line.startswith('DEPTH'):
                    line.replace(' ', '')
                    depth = to_int(line.split('=')[1], i)
                elif line.startswith('LAMBDA'):
                    line.replace(' ', '')
                    l = to_float(line.split('=')[1], i)
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        filename, shutter_speed, *_ = parts
                        filepaths.append(os.path.join(source_dir, filename))
                        exposure_times.append(shutter_speed)
                    else:
                        print(f"Error: Not enough arguments in image_list.txt, Line {i+1}")
                        exit()

        print(f"reading files: {filepaths}")

        images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in filepaths]
        for i, path in enumerate(filepaths):
            if images[i] is None:
                print(f"Error: Can not read file {path}")
                exit()

        assert(len(images) == len(exposure_times))
        lnt = np.log(np.array(exposure_times, dtype=np.float32))

        return (images, lnt, l, alignType, std_img_idx, depth)

    except FileNotFoundError as e:
        print("FileNotFoundError:", e)
        exit()

def ldr_to_channels(images:list[cv2.Mat | np.ndarray[np.uint8, 3]]) -> list[np.ndarray[np.uint8, 3]]:
    channels = [None] * 3
    # channel 0,1,2 = B,G,R
    for i in range(3):
        channels[i] = np.array([img[:, :, i] for img in images], dtype=np.uint8)
    return channels

def read_hdr_image(filepath:str) -> np.ndarray[np.float32, 3]:
    """
    Read the .hdr file and convert it into numpy array with channel order BGR

    Parameters:
    filepath : the path of the .hdr file

    Returns:
    hdr_image[x,y,i] : the HDR value (float32) of pixel location (x, y) in the ith channel (BGR)
    """

    print(f"reading {filepath}")
    hdr_image = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    print(f"size = {hdr_image.shape}")
    return hdr_image

def save_hdr_image(hdr_image:np.ndarray[np.float32, 3], filename:str) -> None:
    """
    Write the numpy array with channel order BGR into .hdr file 

    Parameters:
    filename : the name of the .hdr file to save
    """

    print(f"saving HDR image in {filename}")

    if filename.endswith('.hdr'):
        cv2.imwrite(filename, hdr_image)
    else:
        cv2.imwrite(f"{filename}.hdr", hdr_image)

def process_normalize_type(normalize:str, line:int):
    if normalize == 'NONE' or normalize == '0':
        return NormalizeType.NONE
    elif normalize == 'ALL' or normalize == '1':
        return NormalizeType.ALL
    elif normalize == 'CHANNEL' or normalize == '2':
        return NormalizeType.CHANNEL
    else:
        print(f"Error: NormalizeType {normalize} not found in tonemap.txt, Line {line+1}")
        exit()

def read_tonemap_argument(args:list, line:int) -> list:
    if len(args) < 2:
        print(f"Error: Not enough arguments in tonemap.txt, Line {line+1}")
        exit()

    algorithm = args[0]
    output_dir = args[1]

    if algorithm == 'gamma_intensity' or algorithm == 'gamma_color':
        if len(args) != 5:
            print(f"Error: Need 5 arguments, but {len(args)} in tonemap.txt, Line {line+1}")
            exit()
        gamma = to_float(args[2], line)
        brightness = to_float(args[3], line)
        normalize = process_normalize_type(args[4], line)
        return [algorithm, output_dir, gamma, brightness, normalize]
    elif algorithm == 'global':
        if len(args) != 6:
            print(f"Error: Need 6 arguments, but {len(args)} in tonemap.txt, Line {line+1}")
            exit()
        a = to_float(args[2], line)
        Lwhite = to_float(args[3], line)
        delta = to_float(args[4], line)
        normalize = process_normalize_type(args[5], line)
        return [algorithm, output_dir, a, Lwhite, delta, normalize]
    elif algorithm == 'bilateral':
        if len(args) != 9:
            print(f"Error: Need 9 arguments, but {len(args)} in tonemap.txt, Line {line+1}")
            exit()
        sigma_range = to_float(args[2], line)
        contrast = to_float(args[3], line)
        a = to_float(args[4], line)
        Lwhite = to_float(args[5], line)
        delta = to_float(args[6], line)
        normalize = process_normalize_type(args[7], line)
        save = to_bool(args[8], line)
        return [algorithm, output_dir, sigma_range, contrast, a, Lwhite, delta, normalize, save]
    else:
        print(f"Error: Algorithm name {algorithm} not found in tonemap.txt, Line {line+1}")
        exit()

def read_tonemap_settings(source_dir:str):
    hdr_img = None
    arg_list = []

    try:
        with open(os.path.join(source_dir, 'tonemap.txt'), 'r') as setting:
            for i, line in enumerate(setting):
                line = line.split('#')[0].strip()
                if len(line) == 0:
                    continue
                if line.startswith('FILE'):
                    line.replace(' ', '')
                    hdr_filename = line.split('=')[1]
                    hdr_img = read_hdr_image(os.path.join(source_dir, hdr_filename))
                else:
                    args = line.split()
                    arg_list.append(read_tonemap_argument(args, i))

        return (hdr_img, arg_list)

    except FileNotFoundError as e:
        print("FileNotFoundError:", e)
        exit()
