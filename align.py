import os
import cv2
import numpy as np

NEIGHBORS = np.array([[-1, -1], [ 0, -1], [ 1, -1],
                      [-1,  0], [ 0,  0], [ 1,  0],
                      [-1,  1], [ 0,  1], [ 1,  1]])

def median_threshold(images:list[cv2.Mat | np.ndarray[np.uint8, 3]], save=False) -> cv2.Mat | list[np.ndarray[np.uint8, 2]]:
    binary_imgs = []
    # const = [5, 35, 90, 110]
    # const = [0, 0, 0, 0, 0, 0, 0, 0, 5, 10, 10, 10, 10, 10, 0, 0]
    for i, img in enumerate(images):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thres = np.mean(img)
        # thres = np.median(img)
        # thres = (np.median(img) + np.mean(img)) / 2
        # print(np.median(img), np.mean(img), thres)
        # thres += const[i]
        # _, binary_image = cv2.threshold(gray_img, thres, 255, cv2.THRESH_BINARY)
        _, high = cv2.threshold(gray_img, thres + 5, 255, cv2.THRESH_BINARY)
        _, low = cv2.threshold(gray_img, thres - 5, 255, cv2.THRESH_BINARY)
        different_pixels = np.where(high != low)
        binary_image = high
        binary_image[different_pixels] = 128
        # _, binary_image = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if save:
            cv2.imwrite(f'binary_image_{i+1}.jpg', binary_image)
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

def recursive_find_best_shifts(image:cv2.Mat | np.ndarray[np.uint8, 2], std_img:cv2.Mat | np.ndarray[np.uint8, 2], hierarchy:int) -> tuple[int, int]:
    assert(image.shape == std_img.shape)
    H, W = std_img.shape[:2]

    if hierarchy > 0:
        resized_img = cv2.resize(image, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
        resized_std = cv2.resize(std_img, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
        best_shift = recursive_find_best_shifts(resized_img, resized_std, hierarchy - 1)
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

def mtb(images:list[cv2.Mat | np.ndarray[np.uint8, 3]], std_img_idx:int, hierarchy:int=5, save=False) -> list[cv2.Mat | np.ndarray[np.uint8, 3]]:
    P = len(images)
    assert(std_img_idx >= 0 and std_img_idx < P)
    assert(images[std_img_idx].shape == img.shape for img in images)
    # H, W = std_img.shape[:2]
    bin = median_threshold(images)
    # std_bin = bin[std_img_idx]

    best_shifts = [(0, 0)] * P
    for i in range(std_img_idx - 1, -1, -1):
        best_shifts[i] = recursive_find_best_shifts(bin[i], bin[i + 1], hierarchy)
        best_shifts[i] = tuple_add(best_shifts[i], best_shifts[i + 1])
    for i in range(std_img_idx + 1, P):
        best_shifts[i] = recursive_find_best_shifts(bin[i], bin[i - 1], hierarchy)
        best_shifts[i] = tuple_add(best_shifts[i], best_shifts[i - 1])

    # for i in range(P):
    #     best_shifts[i] = recursive_find_best_shifts(bin[i], bin[std_img_idx], hierarchy)

    print(best_shifts)

    results = []
    for i, img in enumerate(images):
        results.append(translation(img, best_shifts[i][0], best_shifts[i][1]))
        if save:
            cv2.imwrite(f'aligned_image_{i+1}.jpg', results[i])
    return results

def read_ldr_images(source_dir:str):
    filepaths = []
    exposure_times = []

    with open(os.path.join(source_dir, 'image_list.txt'), 'r') as img_list:
        for line in img_list:
            line = line.lstrip()
            if len(line) == 0 or line.startswith('#'):
                continue
            filename, shutter_speed, *_ = line.split()
            filepaths.append(os.path.join(source_dir, filename))
            exposure_times.append(shutter_speed)

    images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in filepaths]
    assert(len(images) == len(exposure_times))

    # lnt = np.log(np.array(exposure_times, dtype=np.float32))
    # print(lnt)

    median_threshold(images, True)
    mtb(images, 2, 5, True)

if __name__ == '__main__':
    read_ldr_images('img/test2')
