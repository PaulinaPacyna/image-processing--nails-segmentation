import glob
import os
import cv2
import numpy as np
import scipy.optimize as op
import itertools
import pandas as pd
import matplotlib.pyplot as plt


def hand_recognition(image):
    lower1 = np.array([0, 30, 80])
    upper1 = np.array([20, 180, 255])
    lower2 = np.array([115, 14, 120])
    upper2 = np.array([180, 50, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_hsv1 = cv2.inRange(hsv, lower1, upper1)
    mask_hsv2 = cv2.inRange(hsv, lower2, upper2)
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
    blurred = cv2.medianBlur(mask_hsv, 5)
    largest = largest_component_mask(blurred)
    return largest, cv2.bitwise_and(image, image, mask=largest)


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


def iou(filename, image):
    if filename == "000.jpg":
        return 0
    test = tests[filename]
    if test.ndim == 3:
        test = cv2.split(test)[0]
    if image.ndim == 3:
        image = cv2.split(image)[0]
    intersection = cv2.bitwise_and(image, test)
    union = cv2.bitwise_or(image, test)
    return round(np.sum(intersection) / np.sum(union), 2)


def dice(filename, image):
    if filename == "000.jpg":
        return 0
    test = tests[filename]
    if test.ndim == 3:
        test = cv2.split(test)[0]
    if image.ndim == 3:
        image = cv2.split(image)[0]
    intersection = cv2.bitwise_and(image, test)
    return round(2 * np.sum(intersection) / (np.sum(image) + np.sum(test)), 2)


if __name__ == "__main__":
    DISPLAY = False
    images = {}
    tests = {}
    for file in glob.glob(os.path.join("nails_segmentation", "images", "*.jpg")):
        # for file in  np.random.choice(glob.glob("*.jpg"), 3):
        images[os.path.basename(file)] = cv2.imread(file)

    for file in glob.glob(os.path.join("nails_segmentation", "labels", "*.jpg")):
        tests[os.path.basename(file)] = cv2.imread(file)
    row = 0
    no_of_rows = 4
    iou_array = []
    dice_array = []

    for filename, image in images.items():
        hand_mask, hand = hand_recognition(image)
        nails_mask, nails = confidence_interval_extraction(hand, hand_mask, 1.5)
        iou_array.append(iou(filename, image))
        dice_array.append(dice(filename, image))
        if DISPLAY:
            images_to_plot = [
                (image, "Original image"),
                (hand, "Extracted hand"),
                (equalization(hand, True), "Hand after equalization"),
                (
                    nails,
                    f"Extracted nails, iou: {iou(filename,image)}, dice: {dice(filename,image)}",
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
