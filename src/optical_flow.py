import cv2
import numpy as np

def compute_sparse_flow(prev_img, next_img):
    """
    Detects key points and computes their movement
    using sparse optical flow.
    """

    # Convert images to grayscale if they are colored
    if len(prev_img.shape) == 3:
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray, next_gray = prev_img, next_img

    # 1. Detect points to track using Shi-Tomasi corner detection
    p0 = cv2.goodFeaturesToTrack(
        prev_gray,
        mask=None,
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # 2. Compute optical flow using the Lucas-Kanade method
    p1, st, err = cv2.calcOpticalFlowPyramidLK(
        prev_gray,
        next_gray,
        p0,
        None
    )

    # Keep only successfully tracked points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    return good_old, good_new