import glob
import os
import cv2
import numpy as np
import unused as un
import scipy.optimize as op


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


def clahe(image, mask=None, hsv=False):
    if hsv == True:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if mask is not None:
        image = random_background(image, mask)
    channels = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(3, 3))
    for i, plane in enumerate(channels):
        channels[i] = clahe.apply(plane)
    image = cv2.merge(channels)
    # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    if mask is not None:
        image = cv2.bitwise_and(image, image, mask=mask)
    if hsv == True:
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def purple(image):
    pass


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


def canny_hsv(carved):
    hsv = cv2.cvtColor(carved, cv2.COLOR_BGR2HSV)
    cnn = cv2.Canny(hsv, 0, 255)
    return cnn


def sat_b(image):
    b = cv2.split(cv2.cvtColor(clahe(image), cv2.COLOR_BGR2LAB))[2]
    s = cv2.split(cv2.cvtColor(clahe(image), cv2.COLOR_BGR2HSV))[1]
    return b + s


def random_background(image, mask):
    rand = np.array(np.round(np.random.rand(*image.shape) * 255), dtype=np.uint8)
    rand = cv2.bitwise_and(rand, rand, mask=cv2.bitwise_not(mask))
    return cv2.add(
        image,
        rand,
    )


def circles(image):
    output = image.copy()
    minDist = int(min(image.shape[:2]) / 10)
    maxRadius = int(min(image.shape[:2]) / 5)
    circles = cv2.HoughCircles(
        image, cv2.HOUGH_GRADIENT, 40, minDist=minDist, maxRadius=maxRadius
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 150, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    return output


def convert(sobelx_64f):
    min = np.min(sobelx_64f)
    sobelx_64f = sobelx_64f - min  # to have only positive values
    max = np.max(sobelx_64f)
    div = max / float(255)
    return np.uint8(np.round(sobelx_64f / div))


def Sobel(gray, k=3):
    ddepth = cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=k)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=k)
    return cv2.convertScaleAbs(cv2.add(gradX, gradY)), cv2.convertScaleAbs(
        cv2.subtract(gradX, gradY)
    )


def mean_extraction(image, error=None, median=False, hsv=False):
    avg = np.median if median else np.mean
    if error is None:
        error = [50, 50, 50]
    if hsv:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    channels = cv2.split(image)
    average = [
        avg(channel.reshape(1, -1)[channel.reshape(1, -1) > 0]) for channel in channels
    ]
    lower, upper = np.array(average) - np.array(error), np.array(average) + np.array(
        error
    )
    mask = cv2.inRange(image, lower, upper)
    if hsv:
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return cv2.medianBlur(mask, 7)


def nails_mean_extraction(image, hand_mask, error=None, median=False, hsv=False):
    nails = cv2.bitwise_not(mean_extraction(image, error, median, hsv))
    return cv2.bitwise_and(nails, hand_mask)


def nails_mean_extraction2(no_nails, hand_mask):
    nails = cv2.bitwise_not(no_nails)
    return cv2.bitwise_and(nails, hand_mask)


def equalization(image, mask=None, hsv=False):
    if hsv == True:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if mask is not None:
        image = random_background(image, mask)
    channels = cv2.split(image)
    for i, plane in enumerate(channels):
        channels[i] = cv2.equalizeHist(plane)
    image = cv2.merge(channels)
    # image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    if mask is not None:
        image = cv2.bitwise_and(image, image, mask=mask)
    # if hsv == True:
    #    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def iou(filename, image):
    if filename == "000.jpg":
        return 0
    test = tests[filename]
    if test.ndim == 3:
        test = cv2.split(test)[0]
    if image.ndim == 3:
        image = cv2.split(image)[0]
    intersection = cv2.bitwise_and(image, test).sum(axis=0).sum()
    union = cv2.bitwise_or(image, test)
    return np.sum(intersection) / np.sum(union)


def coeff():
    components = {}
    hand_masks = {}
    for filename, image in images.items():
        final_mask = largest_component_mask(hsv_mask(image))
        component = cv2.bitwise_and(image, image, mask=final_mask)
        components[filename] = component
        hand_masks[filename] = final_mask

    def mean_iou(error):
        return -np.mean(
            [
                iou(
                    filename,
                    nails_mean_extraction(
                        components[filename],
                        hand_masks[filename],
                        error=error,
                        hsv=True,
                    ),
                )
                for filename in images.keys()
            ]
        )

    opt = op.dual_annealing(mean_iou, ((0, 100), (0, 100), (0, 100)), maxiter=100)
    return opt


def coeff2():
    global result
    components = {}
    hand_masks = {}
    for filename, image in images.items():
        final_mask = largest_component_mask(hsv_mask(image))
        component = cv2.bitwise_and(image, image, mask=final_mask)
        components[filename] = component
        hand_masks[filename] = final_mask

    def mean_iou(error):
        return -np.mean(
            [
                iou(
                    filename,
                    nails_mean_extraction(
                        components[filename],
                        hand_masks[filename],
                        error=error,
                        hsv=True,
                    ),
                )
                for filename in images.keys()
            ]
        )

    ran = [10, 100, 10]
    best = -100000000000000000
    for x in range(*ran):
        print(x)
        for y in range(*ran):

            p = [x, y, 200]
            result = mean_iou(p)
            if best < result:
                best_point = p
                best = result
    return (best_point, result)


def mean_threshold(image, hand, error=1.3, median=True):

    avg = np.median if median else np.mean
    if error is None:
        error = [50, 50, 50]
    channel = cv2.split(image)[1]
    average = avg(channel.reshape(1, -1)[hand.reshape(1, -1) > 0])
    stderr = np.std(channel.reshape(1, -1)[channel.reshape(1, -1) > 0])
    mask = cv2.threshold(
        channel, average - error * stderr, 255, type=cv2.THRESH_BINARY_INV
    )[1]
    hand = cv2.morphologyEx(
        hand,
        cv2.MORPH_ERODE,
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    return cv2.bitwise_and(mask, hand)


def main():
    for filename, image in images.items():
        cv2.namedWindow(
            filename + f"    iou: {iou(filename, image)}", cv2.WINDOW_NORMAL
        )
        final_mask = largest_component_mask(hsv_mask(image))
        component = cv2.bitwise_and(image, image, mask=final_mask)
        clahed_component = equalization(component, mask=final_mask, hsv=True)
        r = min(image.shape[:2]) // 50
        r = r if r % 2 else r + 1
        err = [60, 150, 200]
        cv2.imshow(
            filename + f"    iou: {iou(filename, image)}",
            side_by_side(
                mean_threshold(clahed_component, final_mask),
                nails_mean_extraction(
                    clahed_component, final_mask, error=err, hsv=False
                ),
                nails_mean_extraction(
                    clahed_component, final_mask, error=[30, 200, 200], hsv=False
                ),
                nails_mean_extraction(
                    clahed_component, final_mask, error=err, hsv=False, median=True
                ),
            ),
        ),

        cv2.resizeWindow(filename + f"    iou: {iou(filename, image)}", 600, 600)
        cv2.waitKey(0)


if __name__ == "__main__":
    images = {}
    tests = {}
    for file in glob.glob(os.path.join("nails_segmentation", "images", "*.jpg")):
        # for file in  np.random.choice(glob.glob("*.jpg"), 3):
        images[os.path.basename(file)] = cv2.imread(file)

    for file in glob.glob(os.path.join("nails_segmentation", "labels", "*.jpg")):
        tests[os.path.basename(file)] = cv2.imread(file)
    Tryb = 0
    if Tryb == 0:
        err = [50, 50, 300]
        main()
