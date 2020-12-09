import cv2
import numpy as np

from main import images, test


def hvs_tresholding_error(bl, gl, rl, bu, gu, ru):
    err = 0
    for filename, image in images.items():
        thresholded = cv2.inRange(
            cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
            np.array([bl, gl, rl]),
            np.array([bu, gu, ru]),
        )
        ground_truth, b, c = cv2.split(test[filename])
        err += np.sum((thresholded - ground_truth) ** 2)


def channel(image, channel=0):
    if channel == 0:
        b = image.copy()
        b[:, :, 1] = 0
        b[:, :, 2] = 0
        return b
    if channel == 0:
        g = image.copy()
        g[:, :, 0] = 0
        g[:, :, 2] = 0
        return g
    if channel == 0:
        r = image.copy()
        r[:, :, 0] = 0
        r[:, :, 1] = 0
        return r


def equalization(image):
    b, g, r = cv2.split(image)
    eb = cv2.equalizeHist(b)
    eg = cv2.equalizeHist(g)
    er = cv2.equalizeHist(r)
    return cv2.merge((eb, eg, er))
