import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import utils
import align

def weight_function():
    return np.array([z+1 if z < 128 else 256-z for z in range(256)], dtype=np.float32)

def pick_sample_pixels(H:int, W:int, N:int=400, padding:int=20) -> list[tuple[int, int]]:
    """
    Pick at least 256 pixels in H * W pixels

    Parameters:
    H : height of image
    W : width of image
    N : number of pixels to pick, N >= 256 > (Zmax - Zmin) / (P - 1)
    padding : Areas near the edges of the image where pixels will not be picked

    Returns:
    pixels[i] : the ith of N sample pixel positions
    """

    H -= 2 * padding
    W -= 2 * padding
    assert(N >= 256 and H * W >= N)
    step = H * W // N
    pixels = []
    for i in range(N):
        row = (i * step) // W
        col = (i * step) % W
        assert(row >= 0 and row < H and col >= 0 and col < W)
        pixels.append((padding + row, padding + col))
    return pixels

def images_to_z(images:np.ndarray[np.uint8, 3], sample_pixel_locations:list[tuple[int, int]]) -> np.ndarray[np.uint8, 2]:
    """
    Generate Z from image list and sample pixel locations

    Parameters:
    images[j] : the jth image in list of 2D images
    sample_pixel_pos[i] : (x, y) position of the ith pixel in sample pixels

    Returns:
    Z[i,j] : The pixel value of pixel location i in image j (size=N*P, min=0, max=255)
    """

    P = len(images)
    N = len(sample_pixel_locations)
    Z = np.zeros((N, P), dtype=np.uint8)

    x_coords, y_coords = zip(*sample_pixel_locations)

    for j, img in enumerate(images):
        pixel_values = img[x_coords, y_coords]
        Z[:, j] = pixel_values

    return Z

def solve_response_function(Z:np.ndarray[np.uint8, 2], lnt:np.ndarray[np.float32], l:float, w:np.ndarray[np.float32]) -> tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
    """
    Solve for imaging system response function (camera response function, CRF)

    Solve the least square solution of Debevec matrix equation Ax=b
    A : [N * P + 1 + 254] * [256 + N]
    x : [256 + N] * 1
    b : [N * P + 1 + 254] * 1

    Parameters:
    Z[i,j] : The pixel value of pixel location i in image j (size=N*P, min=0, max=255)
    lnt[j] : The log delta t or log shutter speed for image j (len=P)
    l      : lambda, the constant that determines the amount of smoothness
    w[z]   : weighting function value for pixel value z (len=256)

    Returns:
    g[z]   : log exposure corresponding to pixel value z (len=256)
    lnE[i] : log film irradiance at pixel location i (len=N)
    """

    N, P = Z.shape
    assert(len(lnt) == P and len(w) == 256)
    A = np.zeros((N * P + 1 + 254, 256 + N), dtype=np.float32)
    b = np.zeros((N * P + 1 + 254, 1), dtype=np.float32)
    k = 0
    # Include the data-fitting equations
    for i, pixel in enumerate(Z):
        for j, zij in enumerate(pixel):
            wij = w[zij]
            A[k, zij] = wij
            A[k, 256 + i] = -wij
            b[k, 0] = wij * lnt[j]
            k += 1
    # Fix the curve by setting its middle value to 0
    A[k, 128] = 1
    k += 1
    # Include the smoothness equations
    for i in range(254):
        A[k, i : i + 3] = l * w[i+1] * np.array([1, -2, 1])
        k += 1
    # Solve the system using SVD
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    x = x.ravel()
    return (x[:256], x[256:])

def construct_radiance_map(images:np.ndarray[np.uint8, 3], g:np.ndarray[np.float32], lnt:np.ndarray[np.float32], w:np.ndarray[np.float32]) -> np.ndarray[np.float32]:
    """
    Construct the radiance map (size is same as images)

    Parameters:
    images[i,j,k] : The pixel value of pixel location i,j in image k (len=P, min=0, max=255)
    g[z]          : log exposure corresponding to pixel value z (len=256)
    lnt[j]        : The log delta t or log shutter speed for image j (len=P)
    w[z]          : weighting function value for pixel value z (len=256)

    Returns:
    lnE[z] : log film irradiance at pixel location i (len=N)
    """

    assert(len(images) == len(lnt) and len(g) == 256 and len(w) == 256)
    g_lnt_map = g[images] - lnt[:, np.newaxis, np.newaxis]
    w_map = w[images]
    lnE = np.average(g_lnt_map, axis=0, weights=w_map)
    return lnE

def hdr_reconstruction(channels:list[np.ndarray[np.uint8, 3]], lnt:np.ndarray[np.float32], save=False) -> np.ndarray[np.float32, 3]:
    """
    Read the image_list.txt and read all images included in the list. Then reconstruct those LDR images into a HDR image.

    Parameters:
    channels[i,j,x,y] : the pixel value of pixel location (x, y) in the ith channel of image j
    lnt[j]   : The log delta t or log shutter speed for image j

    Returns:
    hdr_image[x,y,i] : the HDR value (float32) of pixel location (x, y) in the ith channel
    """

    H, W = channels[0][0].shape
    padding = math.ceil(min(H, W) * 0.05)
    pixel_positions = pick_sample_pixels(H, W, padding=padding)
    w = weight_function()
    l = 10

    if save:
        plt.figure(figsize=(10, 10))
        color = ['bx','gx','rx']

    hdr_image = np.zeros((H, W, 3), dtype=np.float32)
    exponential = np.vectorize(lambda x:math.exp(x))

    # channel 0,1,2 = B,G,R
    for i, channel in enumerate(channels):
        # Solve response curves
        Z = images_to_z(channel, pixel_positions)
        g, _ = solve_response_function(Z, lnt, l, w)
        # Construct radiance map
        lnE = construct_radiance_map(channel, g, lnt, w)
        hdr_image[..., i] = exponential(lnE)
        if save:
            plt.plot(g, range(256), color[i])

    if save:
        # Show response curve
        print("save response curve")

        plt.ylabel('pixel value Z')
        plt.xlabel('log exposure X')
        plt.savefig('response-curve.png')

        # Display Radiance map with pseudo-color image (log value)
        print("save radiance map")

        plt.figure(figsize=(12,8))
        plt.imshow(np.log(cv2.cvtColor(hdr_image, cv2.COLOR_BGR2GRAY)), cmap='jet')
        plt.colorbar()
        plt.savefig('radiance-map.png')

        print("save hdr image")

        utils.save_hdr_image(hdr_image, 'hdr')

    return hdr_image

if __name__ == '__main__':

    images, lnt, alignType, std_img_idx = utils.read_ldr_images('img/test2')
    images = align.align(images, alignType, std_img_idx, 5)
    channels = utils.ldr_to_channels(images)
    hdr_reconstruction(channels, lnt, True)
