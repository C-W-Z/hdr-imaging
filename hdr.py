import os
import cv2
import numpy as np

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
