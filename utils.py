import os
import cv2
import numpy as np
from align import AlignType

def read_ldr_images(source_dir:str) -> tuple[list[cv2.Mat | np.ndarray[np.uint8, 3]], np.ndarray[np.float32], AlignType, int]:
    """
    Read the image_list.txt and read all images included in the list. Then converts images into r,g,b channels and log of exposure times

    Parameters:
    source_dir : the path of directory containing image_list.txt and LDR images

    Returns:
    images[j] : the jth image
    lnt[j]    : The log delta t or log shutter speed for image j
    alignType : the alignment method to use
    std_ing_idx : the standard image's index in the list of images (for alignment)
    """

    filepaths = []
    exposure_times = []
    alignType = AlignType.NONE
    std_img_idx = -1

    with open(os.path.join(source_dir, 'image_list.txt'), 'r') as img_list:
        for line in img_list:
            line = line.strip()
            if len(line) == 0 or line.startswith('#'):
                continue
            if line.startswith('ALIGN'):
                line.replace(' ', '')
                t = line.split('=')[1]
                if t == 1 or t == 'OUR':
                    alignType = AlignType.OUR
                elif t == 2 or t == 'CV2':
                    alignType = AlignType.CV2
                elif t != 0 and t != 'NONE':
                    print(f"error: {line}")
            elif line.startswith('STD'):
                line.replace(' ', '')
                std_img_idx = int(line.split('=')[1])
            else:
                filename, shutter_speed, *_ = line.split()
                filepaths.append(os.path.join(source_dir, filename))
                exposure_times.append(shutter_speed)

    print(f"reading files: {filepaths}")

    images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in filepaths]
    assert(len(images) == len(exposure_times))

    lnt = np.log(np.array(exposure_times, dtype=np.float32))

    return (images, lnt, alignType, std_img_idx)

def ldr_to_channels(images:list[cv2.Mat | np.ndarray[np.uint8, 3]]) -> list[np.ndarray[np.uint8, 3]]:
    channels = [None] * 3
    # channel 0,1,2 = B,G,R
    for i in range(3):
        channels[i] = np.array([img[:, :, i] for img in images], dtype=np.uint8)
    return channels

def read_hdr_image(filepath:str) -> np.ndarray[np.float32, 3]:
    """
    Read the .hdr file and convert it into numpy array with channel order BGR

    Parameters:
    filepath : the path of the .hdr file

    Returns:
    hdr_image[x,y,i] : the HDR value (float32) of pixel location (x, y) in the ith channel (BGR)
    """

    print(f"reading {filepath}")
    hdr_image = cv2.imread(filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    print(f"size = {hdr_image.shape}")
    return hdr_image

def save_hdr_image(hdr_image:np.ndarray[np.float32, 3], filename:str) -> None:
    """
    Write the numpy array with channel order BGR into .hdr file 

    Parameters:
    filename : the name of the .hdr file to save
    """

    print("saving HDR image ...")

    cv2.imwrite(f'{filename}.hdr', hdr_image)
