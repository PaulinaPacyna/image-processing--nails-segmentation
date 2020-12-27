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


def largest_component_mask(binary_image: np.array) -> np.array:
    """Select a connected component with the largest area.
    :param binary_image: binary image
    :return: binary image of the largest component"""
    contours = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
        0
    ]
    max_area = 0
    max_contour_index = 0
    for i, contour in enumerate(contours):
        contour_area = cv2.moments(contour)["m00"]
        if contour_area > max_area:
            max_area = contour_area
            max_contour_index = i
    largest_component = np.zeros(binary_image.shape, dtype=np.uint8)
    cv2.drawContours(
        largest_component, contours, max_contour_index, color=255, thickness=-1
    )
    return largest_component


""" 
def equalization(image, hsv=False):
    if hsv == True:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channels = cv2.split(image)
    for i, plane in enumerate(channels):
        channels[i] = cv2.equalizeHist(plane)
    image = cv2.merge(channels)
    if hsv == True:
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image
"""
"""
def confidence_interval_extraction(
    image, hand, eps
):  # equalization, no_clahe_mask, hsv equalization, 1.5, mean ,no_hsv_mean_thresh ,CI
    image = equalization(image, hsv=True)
    lower = [0, 0, 0]
    upper = [0, 0, 0]
    channels = cv2.split(image)
    for i in range(3):
        channel = channels[i]
        average = np.mean(
            channel.reshape(1, -1)[hand.reshape(1, -1) > 0]
        )  # restricting to the hand mask
        stderr = np.std(channel.reshape(1, -1)[hand.reshape(1, -1) > 0])
        lower[i] = average - eps * stderr
        upper[i] = average + eps * stderr
    mask = cv2.bitwise_not(cv2.inRange(image, np.array(lower), np.array(upper)))
    hand = cv2.morphologyEx(
        hand,
        cv2.MORPH_ERODE,
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    nails_mask = cv2.medianBlur(cv2.bitwise_and(mask, hand), 7)
    return nails_mask, cv2.bitwise_and(image, image, mask=nails_mask)
"""


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
    hand = cv2.morphologyEx(
        hand,
        cv2.MORPH_ERODE,
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    nails_mask = cv2.medianBlur(cv2.bitwise_and(mask, hand), 7)

    return nails_mask, cv2.bitwise_and(image, image, mask=nails_mask)


def iou(test: np.array, image: np.array) -> float:
    """
    Compute Intersection over Union coeffiicient.
    :param filename: filename of mask provided in test folder
    :param image: binary mask of nails
    :return: IoU coefficient
    """
    if filename == "000.jpg":
        return 0
    if test.ndim == 3:
        test = cv2.split(test)[0]
    if image.ndim == 3:
        image = cv2.split(image)[0]
    intersection = cv2.bitwise_and(image, test)
    union = cv2.bitwise_or(image, test)
    return np.sum(intersection) / np.sum(union)


def dice(test: np.array, image: np.array) -> float:
    """
    Compute Dice coeffiicient.
    :param filename: filename of mask provided in test folder
    :param image: binary mask of nails
    :return: Dice coefficient
    """
    if filename == "000.jpg":
        return 0
    if test.ndim == 3:
        test = cv2.split(test)[0]
    if image.ndim == 3:
        image = cv2.split(image)[0]
    intersection = cv2.bitwise_and(image, test)
    return 2 * np.sum(intersection) / (np.sum(image) + np.sum(test))


if __name__ == "__main__":
    DISPLAY = 1
    images = {}
    tests = {}
    for file in glob.glob(os.path.join("nails_segmentation", "images", "*.jpg")):
        images[os.path.basename(file)] = cv2.imread(file)

    for file in glob.glob(os.path.join("nails_segmentation", "labels", "*.jpg")):
        tests[os.path.basename(file)] = cv2.imread(file)
    row = 0
    no_of_rows = 4
    iou_array = []
    dice_array = []

    for filename, image in images.items():
        hand_mask, hand = hand_recognition(image)
        nails_mask, nails = saturation_extraction(hand, hand_mask, 1.04)
        saturation = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[1]
        test = tests[filename]
        iou_array.append(iou(test, nails_mask))
        dice_array.append(dice(test, nails_mask))
        if DISPLAY:
            images_to_plot = [
                (image, "Original image"),
                (hand, "Extracted hand"),
                (saturation, "Satueation channel"),
                (
                    nails,
                    f"Extracted nails, iou: {round(iou(test, nails_mask),2)}, dice: {round(dice(test, nails_mask),2)}",
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
