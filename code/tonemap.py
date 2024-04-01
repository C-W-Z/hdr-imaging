import argparse
import math
import os
import cv2
import numpy as np
from enum import IntEnum
import utils

class NormalizeType(IntEnum):
    NONE = 0
    ALL = 1
    CHANNEL = 2
    def __str__(self):
        return self.name.upper()

def to_gray(hdr:np.ndarray):
    if hdr.ndim == 3:
        # channel order is BGR
        # add 1e-6 to avoid zero
        return cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY) + 1e-6
        # return 0.27 * hdr[:,:,2] + 0.67 * hdr[:,:,1] + 0.06 * hdr[:,:,0] + 1e-6
        # return 0.299 * hdr[:,:,2] + 0.587 * hdr[:,:,1] + 0.114 * hdr[:,:,0] + 1e-6
    return hdr

def cv2_drago(hdr:np.ndarray[np.float32, 3], output_dir:str, gamma:float, saturation:float, bias:float, brightness:float):

    print(f"cv2 Drago tonemapping ...\ngamma = {gamma}\nsaturation = {saturation}\nbias = {bias}\nbrightness = {brightness}")

    tonemap = cv2.createTonemapDrago(gamma, saturation, bias)
    ldr = tonemap.process(hdr)
    ldr *= brightness * 255
    filename = f"cv2_drago_{gamma}_{saturation}_{bias}_{brightness}.png"
    filename = os.path.join(output_dir, filename)
    print(f"saving LDR image to {filename}")
    cv2.imwrite(filename, ldr)

def single_global(Lworld:np.ndarray[np.float32, 2], a:float, Lwhite:float, delta:float=1e-6) -> np.ndarray[np.float32, 2]:
    """
    Photographic global mapping for single channel
    """

    H, W = Lworld.shape
    N = H * W
    l_min = np.min(Lworld)
    # print(f"l_min={l_min}")
    if l_min < 0:
        delta += np.abs(l_min)
    average_Lworld = np.exp(np.sum(np.log(delta + Lworld)) / N)
    Lm = a * Lworld / average_Lworld
    Ld = Lm * (1 + Lm / np.square(Lwhite)) / (1 + Lm)
    return Ld

def gamma_intensity(hdr:np.ndarray[np.float32, 3], output_dir:str, gamma:float, brightness:float, normalize:NormalizeType=NormalizeType.ALL):
    print(f"gamma intensity tonemapping ...\ngamma = {gamma}\nbrightness = {brightness}\nnormalize = {normalize}")

    Lworld = to_gray(hdr)
    # print(np.max(Lworld), np.mean(Lworld), np.min(Lworld))
    Ld = np.power(Lworld, 1/gamma)

    ldr = np.zeros_like(hdr)
    for i in range(3):
        ldr[:,:,i] = hdr[:,:,i] * Ld / Lworld
        if normalize == NormalizeType.CHANNEL:
            ldr[:,:,i] = cv2.normalize(ldr[:,:,i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    if normalize == NormalizeType.ALL:
        ldr = cv2.normalize(ldr, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    ldr *= brightness * 255

    filename = f"gamma_intensity_{gamma}_{brightness}_{normalize}.png"
    filename = os.path.join(output_dir, filename)
    print(f"saving LDR image to {filename}")
    cv2.imwrite(filename, ldr)

    return ldr

def gamma_color(hdr:np.ndarray[np.float32, 3], output_dir:str, gamma:float, brightness:float, normalize:NormalizeType=NormalizeType.ALL):
    print(f"gamma color tonemapping ...\ngamma = {gamma}\nbrightness = {brightness}\nnormalize = {normalize}")

    Lworld = to_gray(hdr)
    # print(np.max(Lworld), np.mean(Lworld), np.min(Lworld))
    L = Lworld / (1 / brightness + Lworld)

    ldr = np.zeros_like(hdr)
    for i in range(3):
        ldr[:,:,i] = np.power(hdr[:,:,i] / Lworld, 1/gamma) * L
        if normalize == NormalizeType.CHANNEL:
            ldr[:,:,i] = cv2.normalize(ldr[:,:,i], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    if normalize == NormalizeType.ALL:
        ldr = cv2.normalize(ldr, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    ldr *= brightness * 255

    filename = f"gamma_color_{gamma}_{brightness}_{normalize}.png"
    filename = os.path.join(output_dir, filename)
    print(f"saving LDR image to {filename}")
    cv2.imwrite(filename, ldr)

    return ldr

def photographic_global(hdr:np.ndarray[np.float32, 3], output_dir:str, a:float, Lwhite:float, delta:float, normalize:NormalizeType=NormalizeType.ALL, save_gray:bool=False):
    print(f"phtographic global tonemapping ...\na = {a}\nLwhite = {Lwhite}\ndelta = {delta}\nnormalize = {normalize}")

    Lworld = to_gray(hdr)
    Ld = single_global(Lworld, a, Lwhite, delta)

    if save_gray:
        filename = f"global_gray_{a}_{Lwhite}_{delta}_{normalize}.png"
        filename = os.path.join(output_dir, filename)
        print(f"saving LDR gray image to {filename}")
        cv2.imwrite(filename, Ld * 255)

    ldr = np.zeros_like(hdr, dtype=np.uint8)
    for i in range(3):
        ldr[:,:,i] = single_global(hdr[:,:,i] * Ld / Lworld, a, Lwhite, delta) * 255
        if normalize == NormalizeType.CHANNEL:
            ldr[:,:,i] = cv2.normalize(ldr[:,:,i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    if normalize == NormalizeType.ALL:
        ldr = cv2.normalize(ldr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    filename = f"global_{a}_{Lwhite}_{delta}_{normalize}.png"
    filename = os.path.join(output_dir, filename)
    print(f"saving LDR image to {filename}")
    cv2.imwrite(filename, ldr)

    return ldr

def bilateral_filtering(hdr:np.ndarray[np.float32, 3], output_dir:str, sigma_range:float, contrast:float, a:float, Lwhite:float, delta:float, normalize:NormalizeType=NormalizeType.ALL, save_filtered:bool=False):
    print(f"bilateral filtering tonemapping ...\nsigma_range = {sigma_range}\ncontrast = {contrast}\na = {a}\nLwhite = {Lwhite}\ndelta = {delta}\nnormalize = {normalize}")

    Lworld = to_gray(hdr)
    H, W = Lworld.shape

    gray_img = np.log(Lworld)

    d = np.clip(math.ceil(0.01 * min(H, W)), 5, 11)
    low = cv2.bilateralFilter(gray_img, d, sigma_range, 0.02 * max(H, W))
    high = gray_img - low

    factor = contrast / (np.max(gray_img) - np.min(gray_img))
    log_abs_scale = np.max(gray_img) * factor
    # print(factor, log_abs_scale)

    new_gray = low * factor + high - log_abs_scale

    Ld = np.exp(new_gray)

    if save_filtered:
        low = np.exp(low)
        filename = f"bilateral_low_{sigma_range}_{contrast}_{a}_{Lwhite}_{delta}_{normalize}.png"
        filename = os.path.join(output_dir, filename)
        print(f"saving low frequency image to {filename}")
        cv2.imwrite(filename, single_global(low, a, Lwhite, delta) * 255)

        high = np.exp(high)
        filename = f"bilateral_high_{sigma_range}_{contrast}_{a}_{Lwhite}_{delta}_{normalize}.png"
        filename = os.path.join(output_dir, filename)
        print(f"saving high frequency image to {filename}")
        cv2.imwrite(filename, single_global(high, a, Lwhite, delta) * 255)

        filename = f"bilateral_gray_{sigma_range}_{contrast}_{a}_{Lwhite}_{delta}_{normalize}.png"
        filename = os.path.join(output_dir, filename)
        print(f"saving LDR gray image to {filename}")
        cv2.imwrite(filename, single_global(Ld, a, Lwhite, delta) * 255)

    ldr = np.zeros_like(hdr, dtype=np.uint8)
    for i in range(3):
        # ldr[:,:,i] = hdr[:,:,i] * Ld / Lworld * 255
        ldr[:,:,i] = single_global(hdr[:,:,i] * Ld / Lworld, a, Lwhite, delta) * 255
        if normalize == NormalizeType.CHANNEL:
            ldr[:,:,i] = cv2.normalize(ldr[:,:,i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    if normalize == NormalizeType.ALL:
        ldr = cv2.normalize(ldr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    filename = f"bilateral_{sigma_range}_{contrast}_{a}_{Lwhite}_{delta}_{normalize}.png"
    filename = os.path.join(output_dir, filename)
    print(f"saving LDR image to {filename}")
    cv2.imwrite(filename, ldr)

    return ldr

def main(input_file:str, output_dir:str):
    hdr, arg_list = utils.read_tonemap_settings(input_file)
    for algorithm, *args in arg_list:
        print("")
        if algorithm == 'cv2_drago':
            gamma, saturation, bias, brightness = args
            cv2_drago(hdr, output_dir, gamma, saturation, bias, brightness)
        elif algorithm == 'gamma_intensity':
            gamma, brightness, normalize = args
            gamma_intensity(hdr, output_dir, gamma, brightness, normalize)
        elif algorithm == 'gamma_color':
            gamma, brightness, normalize = args
            gamma_color(hdr, output_dir, gamma, brightness, normalize)
        elif algorithm == 'global':
            a, Lwhite, delta, normalize, save_gray = args
            photographic_global(hdr, output_dir, a, Lwhite, delta, normalize, save_gray)
        elif algorithm == 'bilateral':
            sigma_range, contrast, a, Lwhite, delta, normalize, save_filtered = args
            bilateral_filtering(hdr, output_dir, sigma_range, contrast, a, Lwhite, delta, normalize, save_filtered)
        else:
            print(f"Error: Algorithm name {algorithm} not found")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read HDR image & arguments from information in <input_file> & output the LDR images to <output_directory>\n")
    parser.add_argument("input_file", type=str, help="Input file (.txt) path")
    parser.add_argument("output_directory", type=str, help="Output directory path")

    # usage = parser.format_usage()
    parser.usage = "tonemap.py [-h] <input_file> <output_directory>\n"

    args = parser.parse_args()

    utils.check_and_make_dir(args.output_directory)

    main(args.input_file, args.output_directory)
