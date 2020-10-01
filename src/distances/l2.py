def l2(img1, img2):
    import numpy as np
    import cv2
    sq_dist = cv2.absdiff(img1, img2) ** 2
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        sq_dist = np.sum(sq_dist, axis=-1)
    diff = np.sqrt(sq_dist)
    return diff