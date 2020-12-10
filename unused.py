import cv2
import numpy as np

from src.main import hsv_mask


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
    if channel == 1:
        g = image.copy()
        g[:, :, 0] = 0
        g[:, :, 2] = 0
        return g
    if channel == 2:
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


def find_hand_old(frame):

    YCrCb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    YCrCb_frame = cv2.GaussianBlur(YCrCb_frame, (3, 3), 0)

    mask = cv2.inRange(YCrCb_frame, np.array([0, 127, 75]), np.array([255, 177, 130]))
    bin_mask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.dilate(bin_mask, kernel, iterations=5)
    res = cv2.bitwise_and(frame, frame, mask=bin_mask)

    cv2.imshow("res_old", res)

    return res


def canny_hsv(carved):
    hsv = cv2.cvtColor(carved, cv2.COLOR_BGR2HSV)
    cnn = cv2.Canny(hsv, 0, 255)
    return cnn


def hsv_carving(
    image,
    lower1=None,
    upper1=None,
    lower2=None,
    upper2=None,
    blur=7,
    cl_factor=6,
):
    if upper2 is None:
        upper2 = [180, 50, 255]
    if lower2 is None:
        lower2 = [90, 20, 120]
    if upper1 is None:
        upper1 = [20, 220, 255]
    if lower1 is None:
        lower1 = [0, 30, 80]
    size = image.shape[2:]
    mask_hsv = hsv_mask(image, lower1, lower2, upper1, upper2)
    blurred = cv2.medianBlur(mask_hsv, blur)
    closing_diameter = int(min(size) / cl_factor)
    if closing_diameter > 0:
        closed = cv2.morphologyEx(
            blurred,
            cv2.MORPH_CLOSE,
            kernel=cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (closing_diameter, closing_diameter)
            ),
        )
        carved = cv2.bitwise_and(image, image, mask=cv2.split(closed)[0])
        return carved
    else:
        m = cv2.split(blurred)[0]
        return cv2.bitwise_and(image, image, mask=m)


def yuv_tresholding(image, lower, upper):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    lower = np.array(lower)
    upper = np.array(upper)
    return cv2.cvtColor(cv2.inRange(yuv, lower, upper), cv2.COLOR_GRAY2BGR)


def ycbcr_tresholding(image, lower, upper):
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower = np.array(lower)
    upper = np.array(upper)
    return cv2.cvtColor(cv2.inRange(ycbcr, lower, upper), cv2.COLOR_GRAY2BGR)


def ycrcb_with_morph(image, lower, upper):
    mask = ycbcr_tresholding(image, lower, upper)
    blurred = cv2.medianBlur(mask, 7)
    closing_diameter = int(min(image.shape[:2]) / 6)
    dilation_diameter = int(min(image.shape[:2]) / 15)
    closed = cv2.morphologyEx(
        blurred,
        cv2.MORPH_CLOSE,
        kernel=cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_diameter, closing_diameter)
        ),
    )
    dilated = cv2.morphologyEx(
        closed,
        cv2.MORPH_DILATE,
        kernel=cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_diameter, dilation_diameter)
        ),
    )
    return cv2.bitwise_and(image, image, mask=cv2.split(dilated)[0])


def select_biggest_component(mask):
    mask_ = cv2.split(mask)[0] if mask.ndim == 3 else mask
    cnts = cv2.findContours(
        mask_,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    cnts = cnts[0]
    return sorted(cnts, key=cv2.contourArea, reverse=True)[0]
