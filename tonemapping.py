import utils
import cv2
import numpy as np

def tonemappingDrago(hdr:np.ndarray[np.float32, 3], filename:str):

    print("tonemapping ...")

    tonemap = cv2.createTonemapDrago(0.9, 0.6)
    ldr = tonemap.process(hdr)
    ldr = 2 * ldr
    cv2.imwrite(f"{filename}.png", ldr * 255)

if __name__ == '__main__':
    filename = 'hdr.hdr'
    hdr_image = utils.read_hdr_image(filename)
    tonemappingDrago(hdr_image, 'ldr')
    # hdr.save_hdr_image(hdr_image, 'hdr2')
