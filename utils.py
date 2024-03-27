import os
import cv2
import numpy as np
import OpenEXR
import Imath

def read_ldr_images(source_dir:str, align:bool) -> tuple[np.ndarray[np.uint8, 4], np.ndarray[np.float32]]:
    """
    Read the image_list.txt and read all images included in the list. Then converts images into r,g,b channels and log of exposure times

    Parameters:
    source_dir : the path of directory containing image_list.txt and LDR images

    Returns:
    channels[i,j,x,y] : the pixel value of pixel location (x, y) in the ith channel of image j
    lnt[j]   : The log delta t or log shutter speed for image j
    """

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

    # Align input images based on Median Threshold Bitwise method 
    if align:
        alignMTB = cv2.createAlignMTB()
        alignMTB.process(images, images)

    channels = [None] * 3
    # channel 0,1,2 = B,G,R
    for i in range(3):
        channels[i] = np.array([img[:, :, i] for img in images], dtype=np.uint8)

    lnt = np.log(np.array(exposure_times, dtype=np.float32))

    return (channels, lnt)

def read_hdr_image(filepath:str) -> np.ndarray[np.float32, 3]:
    """
    Read the .hdr file and convert it into numpy array with channel order BGR

    Parameters:
    filepath : the path of the .hdr file

    Returns:
    hdr_image[x,y,i] : the HDR value (float32) of pixel location (x, y) in the ith channel
    """

    exr = OpenEXR.InputFile(filepath)
    header = exr.header()
    dw = header['dataWindow']
    W = dw.max.x - dw.min.x + 1
    H = dw.max.y - dw.min.y + 1
    channels = ['R', 'G', 'B']
    pixels = dict([(c, exr.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))) for c in channels])
    # Convert to numpy array with order BGR
    hdr_image = np.zeros((H, W, len(channels)), dtype=np.float32)
    for i, c in enumerate(channels):
        hdr_image[:, :, 2-i] = np.frombuffer(pixels[c], dtype=np.float32).reshape(H, W)
    return hdr_image

def save_hdr_image(hdr_image:np.ndarray[np.float32, 3], filename:str) -> None:
    """
    Write the numpy array with channel order BGR into .hdr file 

    Parameters:
    filename : the name of the .hdr file to save
    """

    H, W, _ = hdr_image.shape
    header = OpenEXR.Header(W, H)
    float_channel = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict([(c, float_channel) for c in 'RGB'])
    exr = OpenEXR.OutputFile(f"{filename}.hdr", header)
    # Convert hdr image array into bytes
    R = (hdr_image[:,:,2]).astype(np.float32).tobytes()
    G = (hdr_image[:,:,1]).astype(np.float32).tobytes()
    B = (hdr_image[:,:,0]).astype(np.float32).tobytes()
    exr.writePixels({'R': R, 'G': G, 'B': B})
    exr.close()