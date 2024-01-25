import os
import cv2
import numpy as np
import imutils
import xml.etree.ElementTree as ET
from glob import glob

import tqdm


def get_apple_info(img_path: str) -> tuple:
    """Returns a tuple containing (apple_count, apples_bboxes)

    Args:
        img_path (str): Path to the image of apples

    Returns:
        tuple: Returns a tuple containing (apple_count, apples_bboxes)
    """
    xml_path = img_path.replace(".png", ".xml")
    assert os.path.isfile(xml_path), f"File not found: {xml_path}"

    tree = ET.parse(xml_path)
    root = tree.getroot()

    apple_count = 0
    apple_bboxes = []

    for obj_elem in root.findall('.//object'):
        name_elem = obj_elem.find('name')
        bndbox_elem = obj_elem.find('bndbox')

        if name_elem is not None and name_elem.text == 'apple' and bndbox_elem is not None:
            apple_count += 1
            xmin = int(bndbox_elem.find('xmin').text)
            ymin = int(bndbox_elem.find('ymin').text)
            xmax = int(bndbox_elem.find('xmax').text)
            ymax = int(bndbox_elem.find('ymax').text)

            apple_bboxes.append([xmin, ymin, xmax, ymax])

    return apple_count, apple_bboxes


def identify_apples_evaluate_detection(img_path, apple_bboxes):
    original_img = cv2.imread(img_path)

    img = original_img.copy()

    img_h, img_w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    sat_lower = 25
    val_lower = 25

    mask1 = cv2.inRange(hsv, (0, sat_lower, val_lower), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, sat_lower, val_lower), (180, 255, 255))

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 200
    filtered_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    result_img = img.copy()
    cv2.drawContours(result_img, filtered_contours, -1, (0, 255, 0), 2)

    # for apple_bbox in apple_bboxes:
    #     cv2.rectangle(
    #         result_img, apple_bbox[:2], apple_bbox[2:4], (12, 192, 255), 2, cv2.LINE_AA)

    circles = cv2.HoughCircles(
        mask,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=12.5,  # smaller values help detecting cluttered circles
        param1=100,  # higher threshold of Canny edge detector - 300 for cv2.HOUGH_GRADIENT_ALT
        param2=8,  # cv2.HOUGH_GRADIENT - smaller means more false circles | cv2.HOUGH_GRADIENT_ALT - circle perfectness measure
        minRadius=7,
        maxRadius=50)

    # if circles is None:
    #     print("Found 0 circles")
    # else:
    #     print(f"Found {len(np.array(circles).reshape(-1, 3))} circles")

    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        # for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        # cv2.circle(mask, (x, y), r, (0, 0, 255), 4)
        # cv2.rectangle(mask, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    circle_centres = set()

    for circle in circles:
        for bbox_count, bbox in enumerate(apple_bboxes):
            in_box = False

            xmin, ymin, xmax, ymax = bbox

            x_circle = circle[0]
            y_circle = circle[1]

            circle_centres.add((x_circle, y_circle))

            if (ymin <= y_circle <= ymax) and (xmin <= x_circle <= xmax):

                # cv2.circle(img, (x_circle, y_circle),
                #            5, (0, 255, 0), -1, cv2.LINE_AA)
                # cv2.rectangle(img, (xmin, ymin),
                #               (xmax, ymax), (12, 192, 255), 2, cv2.LINE_AA)

                in_box = True

                true_positives += 1

                break

            if not in_box and bbox_count == (len(apple_bboxes) - 1):
                cv2.circle(img, (x_circle, y_circle),
                           5, (0, 0, 255), -1, cv2.LINE_AA)

                false_positives += 1

    for xmin, ymin, xmax, ymax in apple_bboxes:
        for idx, crlc in enumerate(circle_centres):
            not_in = True
            if (xmin <= crlc[0] <= xmax) and (ymin <= crlc[1] <= ymax):
                not_in = False
                break
            if not_in and (idx == len(circle_centres) - 1):

                # cv2.rectangle(img, (xmin, ymin),
                #               (xmax, ymax), (255, 0, 0), 2, cv2.LINE_AA)

                false_negatives += 1

    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f1_score


root_dir = "/home/dexter/Uni/UWE/Machine Vision/Assessment 1/Datasets/Minneapple/detection/train/images/Red/test/"
image_paths = glob(os.path.join(root_dir, '*.png'))

total_precision = 0
total_recall = 0
total_f1_score = 0

for img_path in tqdm.tqdm(image_paths, desc='Processing'):
    _, apple_bboxes = get_apple_info(img_path)
    precision, recall, f1_score = identify_apples_evaluate_detection(
        img_path, apple_bboxes)
    total_precision += precision
    total_recall += recall
    total_f1_score += f1_score

mean_precision = total_precision / len(image_paths)
mean_recall = total_recall / len(image_paths)
mean_f1_score = total_f1_score / len(image_paths)

print(f"Mean Precision: {mean_precision:.4f}")
print(f"Mean Recall: {mean_recall:.4f}")
print(f"Mean F1-Score: {mean_f1_score:.4f}")
