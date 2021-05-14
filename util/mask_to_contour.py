import numpy as np
import cv2
import copy


def mask_to_contour(mask_):
    mask_cp = copy.copy(mask_)
    mask_cp = np.array(np.clip(mask_cp, 0, 255), np.uint8)
    mask_contour, _ = cv2.findContours(mask_cp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_contour = np.array(mask_contour).squeeze()
    return mask_contour
