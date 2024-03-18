import hdr
import cv2
import numpy as np

def tonemappingDrago(hdr:np.ndarray[np.float32, 3], filename:str):
    tonemap = cv2.createTonemapDrago(4, 1.5, 1.5)
    ldr = tonemap.process(hdr)
    cv2.imwrite(f"{filename}.png", ldr * 255)

if __name__ == '__main__':
    filename = 'hdr.hdr'
    hdr_image = hdr.read_hdr_image(filename)
    tonemappingDrago(hdr, 'ldr')
