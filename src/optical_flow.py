import cv2
import numpy as np

def compute_sparse_flow(prev_img, next_img):
    """
    Detects a high density of key points and computes their movement
    using sparse optical flow over strictly consecutive frames.
    """
    if len(prev_img.shape) == 3:
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray, next_gray = prev_img, next_img

    # Modificado para maximizar la densidad estadística pedida por Magri
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        mask=None,
        maxCorners=1200,    # Capacidad masiva de puntos
        qualityLevel=0.005,  # Captura texturas sutiles del asfalto
        minDistance=3,      # Puntos más densos
        blockSize=3
    )
    
    if p0 is None:
        return np.array([]), np.array([])

    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None)

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    return good_old, good_new