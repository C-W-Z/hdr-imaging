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
    Z[i,j] : The pixel value of pixel location i in image j (Z.size=N*P, Z.min=0, Z.max=255)
    lnt[j] : The log delta t or log shutter speed for image j (B.len=P)
    l      : lambda, the constant that determines the amount of smoothness
    w[z]   : weighting function value for pixel value z (w.len=256)

    Returns:
    g[z]   : log exposure corresponding to pixel value z (g.len=256)
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

def construct_radiance_map(Z:np.ndarray[int, 2], g:np.ndarray[float], lnt:np.ndarray[float], w:np.ndarray[float]) -> np.ndarray[float]:
    """
    Construct the radiance map (size is same as images)

    Parameters:
    Z[i,j] : The pixel value of pixel location i in image j (Z.size=N*P, Z.min=0, Z.max=255)
    g[z]   : log exposure corresponding to pixel value z (g.len=256)
    lnt[j] : The log delta t or log shutter speed for image j (B.len=P)
    w[z]   : weighting function value for pixel value z (w.len=256)

    Returns:
    lnE[z] : log film irradiance at pixel location i (len=N)
    """

    N, P = Z.shape
    assert(len(g) == 256 and len(lnt) == P and len(w) == 256)
    lnE = np.zeros((N))
    for i, pixel in enumerate(Z):
        wgtSum = 0
        wSum = 0
        for j, zij in enumerate(pixel):
            wgtSum += w[zij] * (g[zij] - lnt[j])
            wSum += w[zij]
        if wSum > 0:
            lnE[i] = wgtSum / wSum
        else:
            lnE[i] = wgtSum

    return lnE
