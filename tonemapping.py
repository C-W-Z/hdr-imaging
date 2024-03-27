import hdr
import cv2
import numpy as np

def tonemappingDrago(hdr:np.ndarray[np.float32, 3], filename:str):
    tonemap = cv2.createTonemapDrago(1, 0.7)
    ldr = tonemap.process(hdr)
    ldr = 3 * ldr
    cv2.imwrite(f"{filename}.png", ldr * 255)

if __name__ == '__main__':
    filename = 'hdr.hdr'
    hdr_image = hdr.read_hdr_image(filename)
    tonemappingDrago(hdr_image, 'ldr')
    # hdr.save_hdr_image(hdr_image, 'hdr2')
