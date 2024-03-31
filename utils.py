import os
import cv2
import numpy as np
from align import AlignType, ThresholdType
from tonemapping import NormalizeType

def error(message:str):
    print(message)
    exit()

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

def read_ldr_images(source_dir:str) -> tuple[list[cv2.Mat | np.ndarray[np.uint8, 3]], int, np.ndarray[np.float32], AlignType, int, int, ThresholdType, int]:
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
    threshold_type : threshold type in MTB algorithm
    gray_range : the range of intensities to ignore around threshold
    """

    filepaths = []
    exposure_times = []
    l = 20
    align_type = AlignType.NONE
    std_img_idx = -1
    depth = 3
    threshold_type = ThresholdType.MEDIAN
    gray_range = 4

    try:
        with open(os.path.join(source_dir, 'image_list.txt'), 'r') as img_list:
            for i, line in enumerate(img_list):
                line = line.split('#')[0].strip() # remove the comments starts with '#'
                if len(line) == 0:
                    continue

                if line.startswith('ALIGN'):
                    line.replace(' ', '')
                    t = line.split('=')[1]
                    try:
                        align_type = AlignType[t]
                    except KeyError:
                        error(f"Error: {t} is not a valid AlignType, Line {i+1}")

                elif line.startswith('STD'):
                    line.replace(' ', '')
                    std_img_idx = to_int(line.split('=')[1], i)

                elif line.startswith('DEPTH'):
                    line.replace(' ', '')
                    depth = to_int(line.split('=')[1], i)

                elif line.startswith('THRESHOLD'):
                    line.replace(' ', '')
                    t = line.split('=')[1]
                    try:
                        threshold_type = ThresholdType[t]
                    except KeyError:
                        error(f"Error: {t} is not a valid ThresholdType, Line {i+1}")

                elif line.startswith('GRAYRANGE'):
                    line.replace(' ', '')
                    gray_range = to_int(line.split('=')[1], i)

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
                        error(f"Error: Not enough arguments in image_list.txt, Line {i+1}")

        print(f"reading files: {filepaths}")

        images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in filepaths]
        for i, path in enumerate(filepaths):
            if images[i] is None:
                error(f"Error: Can not read file {path}")

        assert(len(images) == len(exposure_times))
        lnt = np.log(np.array(exposure_times, dtype=np.float32))

        return (images, lnt, l, align_type, std_img_idx, depth, threshold_type, gray_range)

    except FileNotFoundError as e:
        error(f"FileNotFoundError:{e}")

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

    print(f"saving HDR image to {filename}")

    if filename.endswith('.hdr'):
        cv2.imwrite(filename, hdr_image)
    else:
        cv2.imwrite(f"{filename}.hdr", hdr_image)

def process_normalize_type(normalize:str, line:int):
    try:
        return NormalizeType[normalize]
    except KeyError:
        error(f"Error: {normalize} is not a valid NormalizeType, Line {line+1}")

def read_tonemap_argument(args:list, line:int) -> list:
    if len(args) < 1:
        error(f"Error: Not enough arguments in tonemap.txt, Line {line+1}")

    algorithm = args[0]

    if algorithm == 'gamma_intensity' or algorithm == 'gamma_color':
        if len(args) != 4:
            error(f"Error: Need 4 arguments, but {len(args)} in tonemap.txt, Line {line+1}")
        gamma = to_float(args[1], line)
        brightness = to_float(args[2], line)
        normalize = process_normalize_type(args[3], line)
        return [algorithm, gamma, brightness, normalize]

    elif algorithm == 'global':
        if len(args) != 6:
            error(f"Error: Need 6 arguments, but {len(args)} in tonemap.txt, Line {line+1}")
        a = to_float(args[1], line)
        Lwhite = to_float(args[2], line)
        delta = to_float(args[3], line)
        normalize = process_normalize_type(args[4], line)
        save_gray = to_bool(args[5], line)
        return [algorithm, a, Lwhite, delta, normalize, save_gray]

    elif algorithm == 'bilateral':
        if len(args) != 8:
            error(f"Error: Need 8 arguments, but {len(args)} in tonemap.txt, Line {line+1}")
        sigma_range = to_float(args[1], line)
        contrast = to_float(args[2], line)
        a = to_float(args[3], line)
        Lwhite = to_float(args[4], line)
        delta = to_float(args[5], line)
        normalize = process_normalize_type(args[6], line)
        save_filtered = to_bool(args[7], line)
        return [algorithm, sigma_range, contrast, a, Lwhite, delta, normalize, save_filtered]

    else:
        error(f"Error: Algorithm name {algorithm} not found in tonemap.txt, Line {line+1}")

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
        error(f"FileNotFoundError:{e}")
