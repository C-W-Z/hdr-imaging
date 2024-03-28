import utils
import cv2
import numpy as np

def to_gray(hdr:np.ndarray):
    if hdr.ndim == 3:
        # channel order is BGR
        # 0.27 * hdr[:,:,2] + 0.67 * hdr[:,:,1] + 0.06 * hdr[:,:,0]
        return 0.299 * hdr[:,:,2] + 0.587 * hdr[:,:,1] + 0.114 * hdr[:,:,0]
    return hdr

def cv2_drago(hdr:np.ndarray[np.float32, 3], multi:float, gamma:float, saturation:float, filename:str):

    print("cv2 Drago tonemapping ...")

    tonemap = cv2.createTonemapDrago(gamma, saturation)
    ldr = tonemap.process(hdr)
    ldr = multi * ldr
    cv2.imwrite(f"{filename}.png", ldr * 255)

def single_global(Lworld:np.ndarray[np.float32, 2], a:float, Lwhite:float):
    H, W = Lworld.shape
    N = H * W
    epsilon = np.abs(np.min(Lworld)) + 1
    average_Lworld = np.exp(np.sum(np.log(epsilon + Lworld)) / N)
    Lm = a * Lworld / average_Lworld
    Ld = Lm * (1 + Lm / np.square(Lwhite)) / (1 + Lm)
    return Ld

def photographic_global(hdr:np.ndarray[np.float32, 3], a:float, Lwhite:float, filename:str):

    print("phtographic global tonemapping ...")

    Lworld = to_gray(hdr)
    Ld = single_global(Lworld, a, Lwhite)

    cv2.imwrite(f"{filename}_gray.png", Ld * 255)

    ldr = np.zeros_like(hdr, dtype=np.uint8)
    for i in range(3):
        ldr[:,:,i] = single_global(hdr[:,:,i] * Ld / Lworld, a, Lwhite) * 255

    cv2.imwrite(f"{filename}.png", ldr)

def bilateral(gray_img:np.ndarray[np.float32, 2], sigma_range:float, contrast:float=np.log(5)):
    H, W = gray_img.shape
    gray_img = np.log(gray_img)
    low = cv2.bilateralFilter(gray_img, -1, sigma_range, 0.02 * max(H, W))
    high = gray_img - low
    factor = contrast / (np.max(gray_img) - np.min(gray_img))
    log_abs_scale = np.max(gray_img) * factor
    print(factor, log_abs_scale)
    new_gray = low * factor + high - log_abs_scale
    return np.exp(new_gray), np.exp(low), np.exp(high)

def bilateral_filtering(hdr:np.ndarray[np.float32, 2], sigma_range:float, contrast:float, a:float, Lwhite:float, filename:str):
    Lworld = to_gray(hdr)
    Ld, low, high = bilateral(Lworld, sigma_range, contrast)
    cv2.imwrite("bilateral_low.png", single_global(low, a, Lwhite) * 255)
    cv2.imwrite("bilateral_high.png", single_global(high, a, Lwhite) * 255)
    cv2.imwrite("bilateral.png", single_global(Ld, a, Lwhite) * 255)

    # factor = contrast / (np.max(np.log(Lworld)) - np.min(np.log(Lworld)))
    # c = np.exp(np.max(low) * factor)

    ldr = np.zeros_like(hdr, dtype=np.uint8)
    for i in range(3):
        # ldr[:,:,i] = hdr[:,:,i] * Ld / Lworld
        # print(np.max(ldr[:,:,i]), np.min(ldr[:,:,i]))
        # ldr[:,:,i] = (ldr[:,:,i]) / (np.max(ldr[:,:,i]) - np.min(ldr[:,:,i])) * 255
        ldr[:,:,i] = single_global(hdr[:,:,i] * Ld / Lworld, a, Lwhite) * 255

    cv2.imwrite(f"{filename}.png", ldr)

if __name__ == '__main__':
    filename = 'hdr.hdr'
    hdr_image = utils.read_hdr_image(filename)
    # cv2_drago(hdr_image, 2, 0.9, 0.6, 'ldr')
    # photographic_global(hdr_image, 2, 100, "ldr")
    bilateral_filtering(hdr_image, 0.4, 5, 10, 100, 'ldr')
