import utils
import cv2
import numpy as np

def cv2Drago(hdr:np.ndarray[np.float32, 3], multi:float, gamma:float, saturation:float, filename:str):

    print("cv2 Drago tonemapping ...")

    tonemap = cv2.createTonemapDrago(gamma, saturation)
    ldr = tonemap.process(hdr)
    ldr = multi * ldr
    cv2.imwrite(f"{filename}.png", ldr * 255)

def photographic_global(hdr:np.ndarray[np.float32, 3], a:float, Lwhite:float, filename:str):

    print("phtographic global tonemapping ...")

    def process(Lworld:np.ndarray[np.float32, 2]):
        average_Lworld = np.exp(np.sum(np.log(1 + Lworld)) / N)
        Lm = a * Lworld / average_Lworld
        Ld = Lm * (1 + Lm / np.square(Lwhite)) / (1 + Lm)
        return Ld

    H, W, _ = hdr.shape
    N = H * W
    # channel order is BGR
    # Lworld = 0.27 * hdr[:,:,2] + 0.67 * hdr[:,:,1] + 0.06 * hdr[:,:,0] + 1e-6
    Lworld = 0.299 * hdr[:,:,2] + 0.587 * hdr[:,:,1] + 0.114 * hdr[:,:,0] + 1e-6

    Ld = process(Lworld)
    cv2.imwrite(f"{filename}_gray.png", Ld * 255)

    ldr = np.zeros_like(hdr, dtype=np.uint8)
    for i in range(3):
        ldr[:,:,i] = process(hdr[:,:,i] * Ld / Lworld) * 255

    cv2.imwrite(f"{filename}.png", ldr)

if __name__ == '__main__':
    filename = 'hdr.hdr'
    hdr_image = utils.read_hdr_image(filename)
    cv2Drago(hdr_image, 2, 0.9, 0.6, 'ldr')
    photographic_global(hdr_image, 2, 100, "ldr")
    # hdr.save_hdr_image(hdr_image, 'hdr2')
