def l1(img1, img2):
    import numpy as np
    import cv2
    diff = cv2.absdiff(img1, img2)
    if img1.shape[-1] == 3 and len(img1.shape) == 3:
        diff = np.sum(diff, axis=-1)
    return diff
