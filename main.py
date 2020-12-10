import glob
import os
import cv2
import numpy as np


def side_by_side(*images):
    def side_by_side_4(a, B, C, D):
        return np.vstack((np.hstack((a, B)), np.hstack((C, D))))

    def side_by_side_9(A, B, C, D, E, F, G, H, I):
        return np.vstack(
            (np.hstack((A, B, C)), np.hstack((D, E, F)), np.hstack((G, H, I)))
        )

    def side_by_side_6(A, B, C, D, E, F):
        return np.vstack((np.hstack((A, B, C)), np.hstack((D, E, F))))

    for i, img in enumerate(images):
        images = list(images)
        if images[i].ndim == 2:
            images[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if len(images) == 1:
        return images[0]
    if len(images) == 2:
        return np.hstack(tuple(images))
    zero = np.zeros(images[0].shape, dtype=np.uint8)
    if len(images) <= 4:
        images = images + [zero]
        return side_by_side_4(*images[:4])
    if len(images) <= 6:
        images = images + [zero]
        return side_by_side_6(*images[:6])
    if len(images) <= 9:
        images = images + [zero] * 2
        return side_by_side_9(*images[:9])


def hsv_tresholding(image, lower, upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array(lower)
    upper = np.array(upper)
    return cv2.inRange(hsv, lower, upper)


def hsv_mask(
    image,
    lower1=None,
    upper1=None,
    lower2=None,
    upper2=None,
    median_blur=5,
    closing_factor=0.1,
):
    if upper2 is None:
        upper2 = [180, 50, 255]
    if lower2 is None:
        lower2 = [115, 14, 120]
    if upper1 is None:
        upper1 = [20, 180, 255]
    if lower1 is None:
        lower1 = [0, 30, 80]
    mask_hsv1 = hsv_tresholding(image, lower1, upper1)
    mask_hsv2 = hsv_tresholding(image, lower2, upper2)
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
    blurred = cv2.medianBlur(mask_hsv, median_blur)

    return mask_hsv


def kmeans(img, K):
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape(img.shape)


def main():
    for filename, image in images.items():
        cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
        final_mask = largest_component_mask(hsv_mask(image))
        closed = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        blurred = cv2.medianBlur(closed, 5)
        component = cv2.bitwise_and(image, image, mask=final_mask)

        cv2.imshow(
            filename,
            side_by_side(image, component, clahe(component)),
        ),

        cv2.resizeWindow(file, 600, 600)
        cv2.waitKey(0)


def largest_component_mask(bin_img):
    contours = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    max_area = 0
    max_contour_index = 0
    for i, contour in enumerate(contours):
        contour_area = cv2.moments(contour)["m00"]
        if contour_area > max_area:
            max_area = contour_area
            max_contour_index = i
    labeled_img = np.zeros(bin_img.shape, dtype=np.uint8)
    cv2.drawContours(labeled_img, contours, max_contour_index, color=255, thickness=-1)
    return labeled_img


os.chdir("./nails_segmentation/images")
images = {}
test = {}
for file in glob.glob("*.jpg"):
    # for file in  np.random.choice(glob.glob("*.jpg"), 3):
    images[file] = cv2.imread(file)
os.chdir("../labels")
for file in glob.glob("*.jpg"):
    test[file] = cv2.imread(file)

main()
