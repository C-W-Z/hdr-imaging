import argparse
import math
import os
import cv2
import numpy as np
from enum import IntEnum
import utils

class AlignType(IntEnum):
    NONE = 0
    OUR = 1
    CV2 = 2
    def __str__(self):
        return self.name.upper()

class ThresholdType(IntEnum):
    MEDIAN_MEAN_AVERAGE = 0
    MEDIAN = 1
    MEAN = 2
    def __str__(self):
        return self.name.upper()

def threshold_bitmap(img:np.ndarray[np.uint8, 3], threshold_type:ThresholdType=ThresholdType.MEDIAN, gray_range:int=4, save_dir:str=None, i:int=0) -> np.ndarray[np.uint8, 2]:

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if threshold_type == ThresholdType.MEDIAN:
        thres = np.median(img)
    elif threshold_type == ThresholdType.MEAN:
        thres = np.mean(img)
    else:
        thres = (np.median(img) + np.mean(img)) / 2

    _, high = cv2.threshold(gray_img, thres + gray_range, 255, cv2.THRESH_BINARY)
    _, low = cv2.threshold(gray_img, thres - gray_range, 255, cv2.THRESH_BINARY)

    different_pixels = np.where(high != low)
    bitmap = high
    bitmap[different_pixels] = 128

    if save_dir != None:
        filename = f"bitmap_{threshold_type}_{gray_range}_{i+1}.jpg"
        filename = os.path.join(save_dir, filename)
        print(f"saving bitmap to {filename}")
        cv2.imwrite(filename, bitmap)

    return bitmap

def img_diff_pixels(img1:np.ndarray[np.uint8, 2], img2:np.ndarray[np.uint8, 2]) -> int:
    assert(img1.shape == img2.shape)
    H, W = img1.shape
    margin = math.ceil(min(H, W) * 0.1)
    center_img1 = img1[margin:-margin, margin:-margin]
    center_img2 = img2[margin:-margin, margin:-margin]
    return H * W - np.count_nonzero((center_img1 == center_img2) & (center_img1 != 128))

def translation(img:np.ndarray[np.uint8, 2], tx:int, ty:int) -> np.ndarray[np.uint8, 2]:
    H, W = img.shape[:2]
    # translation matrix
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    return cv2.warpAffine(img, M, (W, H))

NEIGHBORS = np.array([[-1, -1], [ 0, -1], [ 1, -1],
                      [-1,  0], [ 0,  0], [ 1,  0],
                      [-1,  1], [ 0,  1], [ 1,  1]])

def recursive_find_best_shifts(img:np.ndarray[np.uint8, 3], std_img:np.ndarray[np.uint8, 3], depth:int, threshold_type:ThresholdType, gray_range:int) -> tuple[int, int]:
    assert(img.shape == std_img.shape)
    H, W, *_ = std_img.shape

    if depth > 0:
        resized_img = cv2.resize(img, (W // 2, H // 2))
        resized_std = cv2.resize(std_img, (W // 2, H // 2))
        best_shift = recursive_find_best_shifts(resized_img, resized_std, depth - 1, threshold_type, gray_range)
    else:
        best_shift = (0, 0)

    bin_img = threshold_bitmap(img, threshold_type, gray_range)
    bin_std = threshold_bitmap(std_img, threshold_type, gray_range)

    min_diff = 1e18
    best_delta = (0, 0)
    for tx, ty in NEIGHBORS:
        dx, dy = (tx + best_shift[0] * 2, ty + best_shift[1] * 2)
        shifted = translation(bin_img, dx, dy)
        diff = img_diff_pixels(shifted, bin_std)
        if diff < min_diff:
            min_diff = diff
            best_delta = (dx, dy)
    best_shift = best_delta
    return best_shift

def shift_add(s1:tuple[int, int], s2:tuple[int, int]) -> tuple[int, int]:
    return (s1[0] + s2[0], s1[1] + s2[1])

def mtb(imgs:list[np.ndarray[np.uint8, 3]], std_img_idx:int, depth:int, threshold_type:ThresholdType, gray_range:int, save_dir:str=None) -> list[np.ndarray[np.uint8, 3]]:

    P = len(imgs)
    assert(std_img_idx >= 0 and std_img_idx < P)
    assert(imgs[std_img_idx].shape == img.shape for img in imgs)

    best_shifts = [(0, 0)] * P
    for i in range(std_img_idx - 1, -1, -1):
        best_shifts[i] = recursive_find_best_shifts(imgs[i], imgs[i + 1], depth, threshold_type, gray_range)
        best_shifts[i] = shift_add(best_shifts[i], best_shifts[i + 1])
    for i in range(std_img_idx + 1, P):
        best_shifts[i] = recursive_find_best_shifts(imgs[i], imgs[i - 1], depth, threshold_type, gray_range)
        best_shifts[i] = shift_add(best_shifts[i], best_shifts[i - 1])

    print(f"shifts of images = {best_shifts}")

    results = []
    for i, img in enumerate(imgs):
        results.append(translation(img, best_shifts[i][0], best_shifts[i][1]))
        if save_dir != None:
            filename = f"aligned_{AlignType.OUR}_{std_img_idx}_{depth}_{threshold_type}_{gray_range}-{i+1}.jpg"
            filename = os.path.join(save_dir, filename)
            print(f"saving aligned image to {filename}")
            cv2.imwrite(filename, results[i])

    return results

def align(imgs:list[np.ndarray[np.uint8, 3]], alignType:AlignType, std_img_idx:int, depth:int, threshold_type:ThresholdType, gray_range:int, save_dir:str=None):

    # Align input images based on MTB method
    if alignType == AlignType.CV2:
        print("using cv2 MTB alignment ...")
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(imgs, imgs)
        if save_dir != None:
            for i, img in imgs:
                filename = f"aligned_{AlignType.CV2}_{i+1}.jpg"
                filename = os.path.join(save_dir, filename)
                print(f"saving aligned image to {filename}")
                cv2.imwrite(filename, img)

    elif alignType == AlignType.OUR:
        print("using our MTB alignment ...")
        if std_img_idx < 0 or std_img_idx >= len(imgs):
            std_img_idx = len(imgs) // 2
        imgs = mtb(imgs, std_img_idx, depth, threshold_type, gray_range, save_dir)

    else:
        print(f"AlignType is {AlignType.NONE}")

    return imgs

def main(input_file:str, output_dir:str=None, save_bitmaps:bool=False, save_aligned:bool=False):
    imgs, _, _, alignType, std_img_idx, depth, threshold_type, gray_range = utils.read_ldr_images(input_file)
    align(imgs, alignType, std_img_idx, depth, threshold_type, gray_range, output_dir if save_aligned else None)
    if save_bitmaps:
        for i, img in enumerate(imgs):
            threshold_bitmap(img, threshold_type, gray_range, output_dir, i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Read LDR image & arguments from information in <input_file> & output bitmaps or aligned images to <output_directory>\n")
    parser.add_argument("input_file", type=str, metavar="<input_file>", help="Input file (.txt) path")
    parser.add_argument("-a", action="store_true", help="Output aligned images")
    parser.add_argument("-b", action="store_true", help="Output bitmaps")
    parser.add_argument("-o", type=str, metavar="<output_directory>", help="Output directory path, required if [-a] or [-b]")
    args = parser.parse_args()
    if args.a or args.b:
        if not args.o:
            parser.error("When selecting [-a] or [-b] or both, [-o <output_directory>] is required.")
        utils.check_and_make_dir(args.o)
    main(args.input_file, args.o, args.b, args.a)
