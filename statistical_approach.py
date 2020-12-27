import glob
import os
import cv2
import numpy as np
import scipy.optimize as op
import itertools
import pandas as pd
import matplotlib.pyplot as plt


def hand_recognition(image: np.array) -> (np.array, np.array):
    """Segment hand and corresponding mask from a picture.
    :param image: BGR image
    :return: mask, segmented hand on black background"""
    lower1 = np.array([0, 30, 80])
    upper1 = np.array([20, 180, 255])
    lower2 = np.array([115, 14, 120])
    upper2 = np.array([180, 50, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_hsv1 = cv2.inRange(hsv, lower1, upper1)
    mask_hsv2 = cv2.inRange(hsv, lower2, upper2)
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)  # combining two masks
    blurred = cv2.medianBlur(mask_hsv, 5)
    largest = largest_component_mask(blurred)
    return largest, cv2.bitwise_and(image, image, mask=largest)


def largest_component_mask(image: np.array) -> np.array:
    """Select a connected component with the largest area.
    :param image: binary image
    :return: binary image of the largest component"""
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    max_area = 0
    max_contour_index = 0
    for i, contour in enumerate(contours):
        contour_area = cv2.moments(contour)["m00"]
        if contour_area > max_area:
            max_area = contour_area
            max_contour_index = i
    largest_component = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(
        largest_component, contours, max_contour_index, color=255, thickness=-1
    )
    return largest_component


def saturation_extraction(
    image: np.array, hand: np.array, eps: float
) -> (np.array, np.array):
    """
    Extract nails from an image of hand based on saturation level.
    :param image: BGR image of the hand
    :param hand: mask of the hand
    :param eps: parameter
    :return: binary mask, image of extracted nails
    """
    saturation = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[1]
    average = np.mean(saturation.reshape(-1)[hand.reshape(-1) > 0])
    stderr = np.std(saturation.reshape(-1)[hand.reshape(-1) > 0])
    mask = cv2.threshold(
        saturation, average - eps * stderr, 255, type=cv2.THRESH_BINARY_INV
    )[1]

    nails_mask = cv2.medianBlur(cv2.bitwise_and(mask, hand), 7)

    return nails_mask, cv2.bitwise_and(image, image, mask=nails_mask)


def iou(test: np.array, image: np.array) -> float:
    """
    Compute Intersection over Union coefficient.
    :param test: exemplary image
    :param image: binary mask of nails
    :return: IoU coefficient
    """
    if test.ndim == 3:  # converting BGR picture to one channel
        test = cv2.split(test)[0]
    if image.ndim == 3:
        image = cv2.split(image)[0]
    intersection = cv2.bitwise_and(image, test)
    union = cv2.bitwise_or(image, test)
    return np.sum(intersection) / np.sum(union)


def dice(test: np.array, image: np.array) -> float:
    """
    Compute Dice coefficient.
    :param test: exemplary image
    :param image: binary mask of nails
    :return: Dice coefficient
    """
    if test.ndim == 3:  # converting BGR picture to one channel
        test = cv2.split(test)[0]
    if image.ndim == 3:
        image = cv2.split(image)[0]
    intersection = cv2.bitwise_and(image, test)
    return 2 * np.sum(intersection) / (np.sum(image) + np.sum(test))


if __name__ == "__main__":
    DISPLAY = 0  # if True then pictures of consecutive steps are displayed.
    images = {}  # filename: image (as np.array)
    tests = {}
    for file in glob.glob(os.path.join("nails_segmentation", "images", "*.jpg")):
        images[os.path.basename(file)] = cv2.imread(file)

    for file in glob.glob(os.path.join("nails_segmentation", "labels", "*.jpg")):
        tests[os.path.basename(file)] = cv2.imread(file)
    row = 0  # iterator
    no_of_rows = 4  # number of rows (pictures) to be displayed
    iou_array = []  # array of IoU coefficients
    dice_array = []

    for filename, image in images.items():
        hand_mask, hand = hand_recognition(image)
        nails_mask, nails = saturation_extraction(hand, hand_mask, 1)
        saturation = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[1]
        test = tests[filename]
        iou_array.append(iou(test, nails_mask))
        dice_array.append(dice(test, nails_mask))
        cv2.imwrite(os.path.join("nails_segmentation", "result", filename), hand)
        if DISPLAY:
            images_to_plot = [  # array of tuples - (image, title of image)
                (image, "Original image"),
                (hand, "Extracted hand"),
                (saturation, "Saturation channel"),
                (
                    nails,
                    f"Extracted nails, iou: {round(iou(test, nails_mask), 2)}, dice: {round(dice(test, nails_mask), 2)}",
                ),
                (
                    test,
                    f"Ground truth",
                ),
            ]
            for col, img in enumerate(images_to_plot):
                plt.subplot(
                    no_of_rows, len(images_to_plot), len(images_to_plot) * row + col + 1
                )
                plt.imshow(cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB))
                plt.title(img[1])
                plt.axis("off")
            row += 1
            if row % no_of_rows == 0:
                figManager = plt.get_current_fig_manager()
                figManager.full_screen_toggle()
                plt.draw()
                plt.waitforbuttonpress(0)
                plt.close()
                row -= no_of_rows
    print(
        f"""Average IoU: {np.mean(iou_array)}
Average dice: {np.mean(dice_array)}"""
    )
