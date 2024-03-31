import cv2
import numpy as np
from enum import IntEnum
import utils

class NormalizeType(IntEnum):
    NONE = 0
    ALL = 1
    CHANNEL = 2

def to_gray(hdr:np.ndarray):
    if hdr.ndim == 3:
        # channel order is BGR
        # add 1e-6 to avoid zero
        return cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY) + 1e-6
        # return 0.27 * hdr[:,:,2] + 0.67 * hdr[:,:,1] + 0.06 * hdr[:,:,0] + 1e-6
        # return 0.299 * hdr[:,:,2] + 0.587 * hdr[:,:,1] + 0.114 * hdr[:,:,0] + 1e-6
    return hdr

def normalize(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def cv2_drago(hdr:np.ndarray[np.float32, 3], multi:float, gamma:float, saturation:float, filename:str):

    print("cv2 Drago tonemapping ...")

    tonemap = cv2.createTonemapDrago(gamma, saturation)
    ldr = tonemap.process(hdr)
    ldr = multi * ldr
    cv2.imwrite(f"{filename}.png", ldr * 255)

def aces(color:float):
    A = 2.51
    B = 0.03
    C = 2.43
    D = 0.59
    E = 0.14
    return (color * (A * color + B)) / (color * (C * color + D) + E)

def single_global(Lworld:np.ndarray[np.float32, 2], a:float, Lwhite:float, delta:float=1e-6) -> np.ndarray[np.float32, 2]:
    H, W = Lworld.shape
    N = H * W
    l_min = np.min(Lworld)
    print(f"l_min={l_min}")
    if l_min < 0:
        delta += np.abs(l_min)
    average_Lworld = np.exp(np.sum(np.log(delta + Lworld)) / N)
    Lm = a * Lworld / average_Lworld
    Ld = Lm * (1 + Lm / np.square(Lwhite)) / (1 + Lm)
    return Ld

def gamma_intensity(hdr:np.ndarray[np.float32, 3], gamma:float, brightness:float, normalize:NormalizeType):
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

    return ldr * brightness * 255

def gamma_color(hdr:np.ndarray[np.float32, 3], gamma:float, brightness:float, normalize:NormalizeType):
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

    # cv2.imwrite(f"{filename}.png", ldr * brightness * 255)
    return ldr * brightness * 255

def photographic_global(hdr:np.ndarray[np.float32, 3], a:float, Lwhite:float, delta:float, normalize:NormalizeType):

    print("phtographic global tonemapping ...")

    Lworld = to_gray(hdr)
    Ld = single_global(Lworld, a, Lwhite, delta)

    # cv2.imwrite(f"{filename}_gray.png", Ld * 255)

    ldr = np.zeros_like(hdr, dtype=np.uint8)
    for i in range(3):
        ldr[:,:,i] = single_global(hdr[:,:,i] * Ld / Lworld, a, Lwhite, delta) * 255
        if normalize == NormalizeType.CHANNEL:
            ldr[:,:,i] = cv2.normalize(ldr[:,:,i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    if normalize == NormalizeType.ALL:
        ldr = cv2.normalize(ldr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # cv2.imwrite(f"{filename}.png", ldr)
    return ldr

def bilateral_filtering(hdr:np.ndarray[np.float32, 3], sigma_range:float, contrast:float, a:float, Lwhite:float, delta:float, filename:str):

    Lworld = to_gray(hdr)
    H, W = Lworld.shape

    gray_img = np.log(Lworld)

    # print(0.01 * max(H, W))
    low = cv2.bilateralFilter(gray_img, -1, sigma_range, np.clip(0.01 * max(H, W), 10, 30))
    high = gray_img - low

    factor = contrast / (np.max(gray_img) - np.min(gray_img))
    log_abs_scale = np.max(gray_img) * factor
    print(factor, log_abs_scale)

    new_gray = low * factor + high - log_abs_scale

    low = np.exp(low)
    high = np.exp(high)
    Ld = np.exp(new_gray)

    cv2.imwrite("bilateral_low.png", single_global(low, a, Lwhite, delta) * 255)
    cv2.imwrite("bilateral_high.png", single_global(high, a, Lwhite, delta) * 255)
    cv2.imwrite("bilateral_gray.png", single_global(Ld, a, Lwhite, delta) * 255)

    ldr = np.zeros_like(hdr, dtype=np.uint8)
    for i in range(3):
        # ldr[:,:,i] = hdr[:,:,i] * Ld / Lworld * 255
        ldr[:,:,i] = single_global(hdr[:,:,i] * Ld / Lworld, a, Lwhite, delta) * 255
        if normalize == NormalizeType.CHANNEL:
            ldr[:,:,i] = cv2.normalize(ldr[:,:,i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    if normalize == NormalizeType.ALL:
        ldr = cv2.normalize(ldr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # cv2.imwrite(f"{filename}.png", ldr)
    return ldr

def main(input_dir:str):
    hdr, arg_list = utils.read_tonemap_settings(input_dir)
    for algorithm, output_dir, *args in arg_list:
        if algorithm == 'gamma_intensity':
            gamma, brightness, normalize = args
            gamma_intensity(hdr, gamma, brightness, normalize)

if __name__ == '__main__':

    hdr_image, arg_list = utils.read_tonemap_settings('img/test1')
    print(hdr_image.shape, arg_list)

    # filename = 'img/test1/hdr.hdr'
    # hdr_image = utils.read_hdr_image(filename)
    # npaces = np.vectorize(aces)
    # ldr = npaces(hdr_image)
    # cv2.imwrite("ldr.png", ldr * 255)
    # cv2_drago(hdr_image, 2.8, 1, 0.7, 'ldr')
    # gamma_mapping(hdr_image, 2.2, 5, 'ldr')
    # gamma_map_color(hdr_image, 2.2, 1.2, 'ldr')
    # photographic_global(hdr_image, 1, 50, 0.7, "ldr")
    # bilateral_filtering(hdr_image, 1, 4.5, 4, 25, 0.5, 'ldr')
