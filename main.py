import glob
import os
import cv2
import numpy as np


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


def sidebyside4(A, B, C, D):
    return np.vstack((np.hstack((A, B)), np.hstack((C, D))))


os.chdir('./nails_segmentation/images')
for file in glob.glob('*.jpg'): #np.random.choice(glob.glob("*.jpg"), 3):
    image = cv2.imread(file)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_hsv1 = np.array([0, 40, 80])
    upper_hsv1 = np.array([15, 250, 250])
    lower_hsv2 = np.array([95, 10, 150]) #h,s,v = b,g,r
    upper_hsv2 = np.array([180, 23, 254])
    # lower_hsv2 = np.concatenate([np.array([165]),lower_hsv1[1:]])
    # upper_hsv2 = np.concatenate([np.array([180]),upper_hsv1[1:]])
    mask_hsv1 = cv2.cvtColor(cv2.inRange(hsv, lower_hsv1, upper_hsv1), cv2.COLOR_GRAY2BGR)
    mask_hsv2 = cv2.cvtColor(cv2.inRange(hsv, lower_hsv2, upper_hsv2), cv2.COLOR_GRAY2BGR)
    # lower_image = np.array([10, 10, 10])
    # upper_image = np.array([255, 40, 40])
    # mask_image = cv2.cvtColor(cv2.inRange(image, lower_image, upper_image), cv2.COLOR_GRAY2RGB)
    cv2.namedWindow(file, cv2.WINDOW_NORMAL)
    cv2.imshow(file, sidebyside4(image, hsv, mask_hsv1, mask_hsv2))
    cv2.resizeWindow(file,600,600)
    cv2.waitKey(0)
