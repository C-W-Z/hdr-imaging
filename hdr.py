import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def solve_response_function(Z:np.ndarray[int, 2], lnt:np.ndarray[float], l:float, w:np.ndarray[float]) -> tuple[np.ndarray[float], np.ndarray[float]]:
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
    print(f"N={N},P={P},len(lnt)={len(lnt)},len(w)={len(w)}")
    assert(len(lnt) == P and len(w) == 256)
    A = np.zeros((N * P + 1 + 254, 256 + N))
    b = np.zeros((N * P + 1 + 254, 1))
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
    x, _, _, _ = np.linalg.lstsq(A, b, rcond = None)
    x = x.ravel()
    return (x[:256], x[256:])

def construct_radiance_map(images:np.ndarray[int, 3], g:np.ndarray[float], lnt:np.ndarray[float], w:np.ndarray[float]) -> np.ndarray[float]:
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

def load_exposures(source_dir, channel=0):
    filenames = []
    exposure_times = []
    f = open(os.path.join(source_dir, 'hdr_image_list.txt'))
    for line in f:
        if (line[0] == '#'):
            continue
        (filename, exposure, *rest) = line.split()
        filenames += [filename]
        exposure_times += [exposure]
    
    img_list = [cv2.imread(os.path.join(source_dir, f), 1) for f in filenames]
    img_list = [img[:,:,channel] for img in img_list]
    exposure_times = np.array(exposure_times, dtype=np.float32)

    return (img_list, exposure_times * 16)

def sample_pixels(h, w, x = 20, y = 20):
    ''' 
        Sample pixel positions in a h * w image. 

        Returns a list of tuples representing pixel positions.
    '''
    pos = []
    h_step, w_step = h // (x + 1), w // (y + 1)
    for i in range(1, x + 1):
        for j in range(1, y + 1):
            pos.append((i * h_step, j * w_step))
    return pos

def get_z(images, pixel_positions):
    ''' Images should be a list of 1-channel (R / G / B) images. '''
    h, w = images[0].shape
    z = np.zeros((len(pixel_positions), len(images)), dtype = np.uint8)
    for i, (x, y) in enumerate(pixel_positions):
        for j, img in enumerate(images):
            z[i, j] = img[x, y]
    return z

def hdr2ldr(hdr, filename):
    tonemap = cv2.createTonemapDrago(5)
    ldr = tonemap.process(hdr)
    cv2.imwrite('{}.png'.format(filename), ldr * 255)

if __name__ == '__main__':
    img_dir = 'img/test1'

    print('Reading input images.... ', end='')
    img_list_b, exposure_times = load_exposures(img_dir, 0)
    img_list_g, exposure_times = load_exposures(img_dir, 1)
    img_list_r, exposure_times = load_exposures(img_dir, 2)
    image_height, image_width = img_list_b[0].shape
    print('done')

    # Solving response curves
    print('Solving response curves .... ', end='')
    pixel_positions = sample_pixels(image_height, image_width)
    w = np.array([z if z < 128 else 255-z for z in range(256)])
    l = 10
    Zb = get_z(img_list_b, pixel_positions)
    Zg = get_z(img_list_g, pixel_positions)
    Zr = get_z(img_list_r, pixel_positions)
    lnt = np.array([math.log(e,2) for e in exposure_times])
    gb, _ = solve_response_function(Zb, lnt, l, w)
    gg, _ = solve_response_function(Zg, lnt, l, w)
    gr, _ = solve_response_function(Zr, lnt, l, w)
    print('done')

    # Show response curve
    print('Saving response curves plot .... ', end='')
    plt.figure(figsize=(10, 10))
    plt.plot(gr, range(256), 'rx')
    plt.plot(gg, range(256), 'gx')
    plt.plot(gb, range(256), 'bx')
    plt.ylabel('pixel value Z')
    plt.xlabel('log exposure X')
    plt.savefig('response-curve.png')
    print('done')

    print('Constructing HDR image: ')
    hdr = np.zeros((len(img_list_b[0]), len(img_list_b[0][0]), 3), 'float32')
    vfunc = np.vectorize(lambda x:math.exp(x))
    E = construct_radiance_map(img_list_b, gb, lnt, w)
    hdr[..., 0] = vfunc(E)
    E = construct_radiance_map(img_list_g, gg, lnt, w)
    hdr[..., 1] = vfunc(E)
    E = construct_radiance_map(img_list_r, gr, lnt, w)
    hdr[..., 2] = vfunc(E)
    print('done')

    # Display Radiance map with pseudo-color image (log value)
    print('Saving pseudo-color radiance map .... ', end='')
    plt.figure(figsize=(12,8))
    plt.imshow(np.log2(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)), cmap='jet')
    plt.colorbar()
    plt.savefig('radiance-map.png')
    print('done')

    print('Saving LDR image .... ', end='')
    hdr2ldr(hdr, 'ldr')
    print('done')
