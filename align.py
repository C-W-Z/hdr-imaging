import cv2
import numpy as np
from enum import IntEnum
import utils

class AlignType(IntEnum):
    NONE = 0
    OUR = 1
    CV2 = 2
    def __str__(self):
        return self.name.capitalize()

NEIGHBORS = np.array([[-1, -1], [ 0, -1], [ 1, -1],
                      [-1,  0], [ 0,  0], [ 1,  0],
                      [-1,  1], [ 0,  1], [ 1,  1]])

def median_threshold(images:list[cv2.Mat | np.ndarray[np.uint8, 3]], save=False) -> cv2.Mat | list[np.ndarray[np.uint8, 2]]:
    binary_imgs = []
    for i, img in enumerate(images):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # thres = np.mean(img)
        # thres = np.median(img)
        thres = (np.median(img) + np.mean(img)) / 2
        # print(np.median(img), np.mean(img), thres)
        # _, binary_image = cv2.threshold(gray_img, thres, 255, cv2.THRESH_BINARY)
        # _, binary_image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, high = cv2.threshold(gray_img, thres + 4, 255, cv2.THRESH_BINARY)
        _, low = cv2.threshold(gray_img, thres - 4, 255, cv2.THRESH_BINARY)
        different_pixels = np.where(high != low)
        binary_image = high
        binary_image[different_pixels] = 128
        if save:
            cv2.imwrite(f'bitmap_{i+1}.jpg', binary_image)
        binary_imgs.append(binary_image)
    return binary_imgs

def img_diff_pixels(img1:cv2.Mat | np.ndarray[np.uint8, 2], img2:cv2.Mat | np.ndarray[np.uint8, 2]) -> int:
    assert(img1.shape == img2.shape)
    # return np.count_nonzero((img1 != img2))
    return np.count_nonzero((img1 != img2) & (img1 != 128) & (img2 != 128))
    # return np.sum(np.abs(img1.astype(np.int32) - img2.astype(np.int32)))

def translation(image:cv2.Mat | np.ndarray[np.uint8, 2], tx:int, ty:int) -> cv2.Mat | np.ndarray[np.uint8, 2]:
    H, W = image.shape[:2]
    max_x = max(W, W + tx)
    max_y = max(H, H + ty)
    extended_image = np.full((max_y, max_x) + image.shape[2:], 128, dtype=image.dtype)

    start_x = max(0, tx)
    start_y = max(0, ty)
    extended_image[start_y:start_y + H, start_x:start_x + W] = image

    # translation matrix
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    result = cv2.warpAffine(image, M, (W, H))
    return result

def tuple_add(tuple1:tuple[int, int], tuple2:tuple[int, int]) -> tuple[int, int]:
    return (tuple1[0] + tuple2[0], tuple1[1] + tuple2[1])

def recursive_find_best_shifts(image:cv2.Mat | np.ndarray[np.uint8, 2], std_img:cv2.Mat | np.ndarray[np.uint8, 2], depth:int) -> tuple[int, int]:
    assert(image.shape == std_img.shape)
    H, W = std_img.shape[:2]

    if depth > 0:
        resized_img = cv2.resize(image, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
        resized_std = cv2.resize(std_img, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
        best_shift = recursive_find_best_shifts(resized_img, resized_std, depth - 1)
    else:
        best_shift = (0, 0)

    min_diff = 1e18
    best_delta = (0, 0)
    for tx, ty in NEIGHBORS:
        delta = (tx + best_shift[0] * 2, ty + best_shift[1] * 2)
        shifted = translation(image, delta[0], delta[1])
        diff = img_diff_pixels(shifted, std_img)
        if diff < min_diff:
            min_diff = diff
            best_delta = delta
    best_shift = best_delta
    return best_shift

def mtb(images:list[cv2.Mat | np.ndarray[np.uint8, 3]], std_img_idx:int, depth:int=5) -> list[cv2.Mat | np.ndarray[np.uint8, 3]]:
    P = len(images)
    assert(std_img_idx >= 0 and std_img_idx < P)
    assert(images[std_img_idx].shape == img.shape for img in images)
    bin = median_threshold(images)

    best_shifts = [(0, 0)] * P
    for i in range(std_img_idx - 1, -1, -1):
        best_shifts[i] = recursive_find_best_shifts(bin[i], bin[i + 1], depth)
        best_shifts[i] = tuple_add(best_shifts[i], best_shifts[i + 1])
    for i in range(std_img_idx + 1, P):
        best_shifts[i] = recursive_find_best_shifts(bin[i], bin[i - 1], depth)
        best_shifts[i] = tuple_add(best_shifts[i], best_shifts[i - 1])

    # for i in range(P):
    #     best_shifts[i] = recursive_find_best_shifts(bin[i], bin[std_img_idx], depth)

    print(f"image shifts = {best_shifts}")

    results = []
    for i, img in enumerate(images):
        results.append(translation(img, best_shifts[i][0], best_shifts[i][1]))
        # if save:
        #     cv2.imwrite(f'aligned_image_{i+1}.jpg', results[i])
    return results

def align(images:list[cv2.Mat | np.ndarray[np.uint8, 3]], alignType:AlignType, std_img_idx:int=-1, depth:int=5):

    # Align input images based on Median Threshold Bitwise method
    if alignType == AlignType.CV2:
        print("using cv2 MTB alignment ...")
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(images, images)
    elif alignType == AlignType.OUR:
        print("using our MTB alignment ...")
        if std_img_idx < 0:
            std_img_idx = len(images) // 2
        images = mtb(images, std_img_idx, depth)

    return images

if __name__ == '__main__':

    images, lnt, _, alignType, std_img_idx, depth = utils.read_ldr_images('img/test3')
    median_threshold(images, save=True)
    images = align(images, alignType, std_img_idx, depth)
