import glob
import os
import cv2
import numpy as np


def side_by_side_4(a, B, C, D):
    return np.vstack((np.hstack((a, B)), np.hstack((C, D))))


def side_by_side_6(A, B, C, D, E, F):
    return np.vstack((np.hstack((A, B)), np.hstack((C, D)), np.hstack((E, F))))


def hsv_tresholding(image, lower, upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv1 = np.array(lower)
    upper_hsv1 = np.array(upper)
    return cv2.cvtColor(cv2.inRange(hsv, lower_hsv1, upper_hsv1), cv2.COLOR_GRAY2BGR)


def smth():
    for filename, image in images.items():
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_hsv1 = hsv_tresholding(image, [0, 25, 80], [20, 220, 250])
        blurred = cv2.medianBlur(mask_hsv1, 5)
        opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel=np.ones((5, 5)))
        cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
        cv2.imshow(
            filename,
            side_by_side_6(
                image,
                hsv,
                mask_hsv1,
                blurred,
                cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel=np.ones((5, 5))),
                cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel=np.ones((21, 21))),
            ),
        )
        # cv2.imshow(file, side_by_side_4(image, hsv, mask_hsv1, cv2.bitwise_or(mask_hsv1, mask_hsv2)))
        cv2.resizeWindow(file, 600, 600)
        cv2.waitKey(0)


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
