import glob
import os
import cv2
import numpy as np


def side_by_side_4(a, B, C, D):
    return np.vstack((np.hstack((a, B)), np.hstack((C, D))))


def side_by_side_9(A, B, C, D, E, F, G, H, I):
    return np.vstack((np.hstack((A, B, C)), np.hstack((D, E, F)), np.hstack((G, H, I))))


def side_by_side_6(A, B, C, D, E, F):
    return np.hstack((np.vstack((A, B)), np.vstack((C, D)), np.vstack((E, F))))


def hsv_tresholding(image, lower, upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(lower)
    upper = np.array(upper)
    return cv2.cvtColor(cv2.inRange(hsv, lower, upper), cv2.COLOR_GRAY2BGR)


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


def clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    return cv2.cvtColor(lab, cv2.COLOR_YUV2BGR)


def ycrcb_with_morph(image, lower, upper):
    size = image.shape[:2]
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


def side_by_side(*images):
    for i, img in enumerate(images):
        images = list(images)
        if images[i].ndim == 2:
            images[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if len(images) == 1:
        return images[0]
    if len(images) == 2:
        return (np.hstack(tuple(images)))
    zero = np.zeros(images[0].shape,dtype=np.uint8)
    if len(images) <= 4:
        images = images + [zero]
        return side_by_side_4(*images[:4])
    if len(images) <= 6:
        images = images + [zero]
        return side_by_side_6(*images[:6])
    if len(images) <= 9:
        images = images + [zero] * 2
        return side_by_side_9(*images[:9])


def hsv_carving(image):
    size = image.shape[:2]
    mask_hsv1 = hsv_tresholding(image, [0, 30, 80], [20, 220, 255])
    mask_hsv2 = hsv_tresholding(image, [90, 20, 120], [180, 50, 255])
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
    blurred = cv2.medianBlur(mask_hsv, 7)
    closing_diameter = int(min(size) / 6)
    dilation_diameter = int(min(size) / 15)
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
    carved = cv2.bitwise_and(image, image, mask=cv2.split(dilated)[0])
    return carved


def find_hand_old(frame):
    img = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    YCrCb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # YCrCb_frame = cv2.GaussianBlur(YCrCb_frame, (9, 9), 0)
    YCrCb_frame = cv2.GaussianBlur(YCrCb_frame, (3, 3), 0)
    # YCrCb_frame = cv2.GaussianBlur(YCrCb_frame, (1, 1), 0)
    # cv2.imshow("YCrCb_frame_old", YCrCb_frame)
    # print(frame.shape[:2])
    # mask = cv2.inRange(YCrCb_frame, np.array([0, 135, 97]), np.array([255, 177, 127]))#140 170 100 120
    # mask = cv2.inRange(YCrCb_frame, np.array([0, 133, 77]), np.array([255, 173, 127])) # best enough
    mask = cv2.inRange(YCrCb_frame, np.array([0, 127, 75]), np.array([255, 177, 130]))
    bin_mask = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bin_mask = cv2.dilate(bin_mask, kernel, iterations=5)
    res = cv2.bitwise_and(frame, frame, mask=bin_mask)

    cv2.imshow("res_old", res)

    return res


def smth():
    for filename, image in images.items():
        image = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        carved = hsv_carving(image)
        cnn = edges(carved)
        cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
        h,s,v=cv2.split(hsv)
        by_side = side_by_side(image, carved, s)
        cv2.imshow(
            filename,
            by_side
        )
        cv2.resizeWindow(file, 600, 600)
        cv2.waitKey(0)


def edges(carved):
    hsv = cv2. cvtColor(carved, cv2.COLOR_BGR2HSV)
    cnn = cv2.Canny(hsv, 0, 255)
    return cnn


images = {}
test = {}

os.chdir("./nails_segmentation/images")
for file in glob.glob("*.jpg"):
    # for file in  np.random.choice(glob.glob("*.jpg"), 3):
    images[file] = cv2.imread(file)
os.chdir("../labels")
for file in glob.glob("*.jpg"):
    test[file] = cv2.imread(file)

smth()
