import os
import cv2
import numpy as np
from align import AlignType, ThresholdType
from tonemap import NormalizeType

def perror(message:str):
    print(message)
    exit()

def to_bool(value:str, file:str, line:int):
    if value.capitalize() == 'True' or value == '1':
        return True
    elif value.capitalize() == 'False' or value == '0':
        return False
    else:
        perror(f"Error in {file}, line {line+1}: {value} is not a valid bool value")

def to_int(value:str, file:str, line:int):
    try:
        return int(value)
    except ValueError:
        perror(f"Error in {file}, line {line+1}: {value} is not a valid int value")

def to_float(value:str, file:str, line:int):
    try:
        return float(value)
    except ValueError:
        perror(f"Error in {file}, line {line+1}: {value} is not a valid float value")

def to_AlignType(value:str, file:str, line:int):
    try:
        return AlignType[value]
    except KeyError:
        perror(f"Error in {file}, line {line+1}: {value} is not a valid AlignType value")

def to_ThresholdType(value:str, file:str, line:int):
    try:
        return ThresholdType[value]
    except KeyError:
        perror(f"Error in {file}, line {line+1}: {value} is not a valid ThresholdType value")

def to_NormalizeType(value:str, file:str, line:int):
    try:
        return NormalizeType[value]
    except KeyError:
        perror(f"Error in {file}, line {line+1}: {value} is not a valid NormalizeType value")

def read_ldr_images(image_list:str) -> tuple[list[cv2.Mat | np.ndarray[np.uint8, 3]], int, np.ndarray[np.float32], AlignType, int, int, ThresholdType, int]:
    """
    Read the image_list file (.txt) and read all images included in the list. Then converts images into r,g,b channels and log of exposure times

    Parameters:
    image_list : the path of image_list.txt

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

    images = []
    exposure_times = []
    l = 20
    align_type = AlignType.NONE
    std_img_idx = -1
    depth = 3
    threshold_type = ThresholdType.MEDIAN
    gray_range = 4

    input_dir = os.path.dirname(image_list)

    try:
        with open(image_list, 'r') as img_list:
            for i, line in enumerate(img_list):
                line = line.split('#')[0].strip() # remove the comments starts with '#'
                if len(line) == 0:
                    continue

                if line.startswith('ALIGN'):
                    line.replace(' ', '')
                    t = line.split('=')[1]
                    align_type = to_AlignType(t, image_list, i)

                elif line.startswith('STD'):
                    line.replace(' ', '')
                    std_img_idx = to_int(line.split('=')[1], image_list, i)

                elif line.startswith('DEPTH'):
                    line.replace(' ', '')
                    depth = to_int(line.split('=')[1], image_list, i)

                elif line.startswith('THRESHOLD'):
                    line.replace(' ', '')
                    t = line.split('=')[1]
                    threshold_type = to_ThresholdType(t, image_list, i)

                elif line.startswith('GRAYRANGE'):
                    line.replace(' ', '')
                    gray_range = to_int(line.split('=')[1], image_list, i)

                elif line.startswith('LAMBDA'):
                    line.replace(' ', '')
                    l = to_float(line.split('=')[1], image_list, i)

                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        filename, shutter_speed, *_ = parts
                        filepath = os.path.join(input_dir, filename)
                        print(f"reading file {filepath}")
                        img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                        if img is None:
                            perror(f"Error: Can not read file {filepath}")
                        images.append(img)
                        exposure_times.append(shutter_speed)
                    else:
                        perror(f"Error in {image_list}, line {i+1}: Not enough arguments")

        assert(len(images) == len(exposure_times))
        lnt = np.log(np.array(exposure_times, dtype=np.float32))

        return (images, lnt, l, align_type, std_img_idx, depth, threshold_type, gray_range)

    except FileNotFoundError as e:
        perror(f"FileNotFoundError: {e}")

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
    if hdr_image is None:
        perror(f"Error: Can not read file {filepath}")
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

def read_tonemap_arguments(args:list, file:str, line:int) -> list:
    if len(args) < 1:
        perror(f"Error in {file}, line {line+1}: Not enough arguments")

    algorithm = args[0]

    if algorithm == 'cv2_drago':
        if len(args) != 5:
            perror(f"Error in {file}, line {line+1}: Need 5 arguments, but got {len(args)}")
        gamma = to_float(args[1], file, line)
        saturation = to_float(args[2], file, line)
        bias = to_float(args[3], file, line)
        brightness = to_float(args[4], file, line)
        return [algorithm, gamma, saturation, bias, brightness]

    elif algorithm == 'gamma_intensity' or algorithm == 'gamma_color':
        if len(args) != 4:
            perror(f"Error in {file}, line {line+1}: Need 4 arguments, but got {len(args)}")
        gamma = to_float(args[1], file, line)
        brightness = to_float(args[2], file, line)
        normalize = to_NormalizeType(args[3], file, line)
        return [algorithm, gamma, brightness, normalize]

    elif algorithm == 'global':
        if len(args) != 6:
            perror(f"Error in {file}, line {line+1}: Need 6 arguments, but got {len(args)}")
        a = to_float(args[1], file, line)
        Lwhite = to_float(args[2], file, line)
        delta = to_float(args[3], file, line)
        normalize = to_NormalizeType(args[4], file, line)
        save_gray = to_bool(args[5], file, line)
        return [algorithm, a, Lwhite, delta, normalize, save_gray]

    elif algorithm == 'bilateral':
        if len(args) != 8:
            perror(f"Error in {file}, line {line+1}: Need 8 arguments, but got {len(args)}")
        sigma_range = to_float(args[1], file, line)
        contrast = to_float(args[2], file, line)
        a = to_float(args[3], file, line)
        Lwhite = to_float(args[4], file, line)
        delta = to_float(args[5], file, line)
        normalize = to_NormalizeType(args[6], file, line)
        save_filtered = to_bool(args[7], file, line)
        return [algorithm, sigma_range, contrast, a, Lwhite, delta, normalize, save_filtered]

    else:
        perror(f"Error in {file}, line {line+1}: Algorithm name {algorithm} not found")

def read_tonemap_settings(setting_file:str) -> tuple[np.ndarray[np.float32, 3],list]:
    hdr_img = None
    arg_list = []

    input_dir = os.path.dirname(setting_file)

    try:
        with open(setting_file, 'r') as setting:
            for i, line in enumerate(setting):
                line = line.split('#')[0].strip() # remove the comments starts with '#'
                if len(line) == 0:
                    continue
                if line.startswith('FILE'):
                    line.replace(' ', '')
                    hdr_filename = line.split('=')[1]
                    hdr_img = read_hdr_image(os.path.join(input_dir, hdr_filename))
                else:
                    args = line.split()
                    arg_list.append(read_tonemap_arguments(args, setting_file, i))

        return (hdr_img, arg_list)

    except FileNotFoundError as e:
        perror(f"FileNotFoundError: {e}")

def check_and_make_dir(dir:str):
    if os.path.isdir(dir):
        return
    print(f"{dir} is not a directory")
    try:
        print(f"Making directory {dir} ...")
        os.makedirs(dir, exist_ok=True)
        print(f"Success")
    except Exception as e:
        perror(f"Error: {e}")
