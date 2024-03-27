import os
import cv2
import numpy as np

NEIGHBORS = np.array([[-1, -1], [ 0, -1], [ 1, -1],
                      [-1,  0], [ 0,  0], [ 1,  0],
                      [-1,  1], [ 0,  1], [ 1,  1]])

def median_threshold(images:list[cv2.Mat | np.ndarray[np.uint8, 3]], save=False) -> cv2.Mat | list[np.ndarray[np.uint8, 2]]:
    binary_imgs = []
    for i, img in enumerate(images):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thres = np.mean(img)
        _, binary_img = cv2.threshold(gray_img, thres, 255, cv2.THRESH_BINARY)
        if save:
            cv2.imwrite(f'binary_image_{i+1}.jpg', binary_img)
        binary_imgs.append(binary_img)
    return binary_imgs

def img_diff_pixels(img1:cv2.Mat | np.ndarray[np.uint8, 2], img2:cv2.Mat | np.ndarray[np.uint8, 2]) -> int:
    assert(img1.shape == img2.shape)
    return np.count_nonzero(img1 != img2)

def translation(image:cv2.Mat | np.ndarray[np.uint8, 2], tx:int, ty:int) -> cv2.Mat | np.ndarray[np.uint8, 2]:
    H, W = image.shape[:2]
    # translation matrix
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    result = cv2.warpAffine(image, M, (W, H))
    return result

def recursive_find_best_shifts(images:list[cv2.Mat | np.ndarray[np.uint8, 2]], std_img:cv2.Mat | np.ndarray[np.uint8, 2], hierarchy:int) -> list[tuple[int, int]]:
    assert(len(images) > 1)
    assert(std_img.shape == img.shape for img in images)

    H, W = std_img.shape[:2]

    if hierarchy > 0:
        resized_images = [cv2.resize(img, (W // 2, H // 2)) for img in images]
        resized_std = cv2.resize(std_img, (W // 2, H // 2))
        best_shifts = recursive_find_best_shifts(resized_images, resized_std, hierarchy - 1)
    else:
        best_shifts = [(0, 0)] * len(images)

    for i, img in enumerate(images):
        min_diff = 1e18
        best_shift = (0, 0)
        for tx, ty in NEIGHBORS:
            delta = (tx + best_shifts[i][0] * 2, ty + best_shifts[i][1] * 2)
            shifted = translation(img, delta[0], delta[1])
            diff = img_diff_pixels(shifted, std_img)
            if diff < min_diff:
                min_diff = diff
                best_shift = delta
        best_shifts[i] = best_shift
    return best_shifts

def mtb(images:list[cv2.Mat | np.ndarray[np.uint8, 3]], std_img:cv2.Mat | np.ndarray[np.uint8, 3], hierarchy:int=5, save=False) -> list[cv2.Mat | np.ndarray[np.uint8, 3]]:
    assert(len(images) > 1)
    assert(std_img.shape == img.shape for img in images)
    H, W = std_img.shape[:2]
    binary_imags = median_threshold(images)
    std_binary = median_threshold([std_img])[0]

    best_shifts = recursive_find_best_shifts(binary_imags, std_binary, hierarchy)
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

    mtb(images, images[0], 6, True)

if __name__ == '__main__':
    read_ldr_images('img/test2')
