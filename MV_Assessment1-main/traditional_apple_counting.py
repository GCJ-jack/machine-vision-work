import os
import cv2
import numpy as np
import imutils
import xml.etree.ElementTree as ET
from glob import glob

import tqdm
from matplotlib import pyplot as plt
from scipy import ndimage

image_dir = "/Users/guochaojun/Desktop/MV_Assessment1_dataset/test"
# dataset = AppleDataset(root_dir=image_dir, transforms=None)

total_true_positive = 0
total_false_positive = 0
total_false_negative = 0

image_paths = glob(os.path.join(image_dir, '*.png'))

red_bounds_one = np.array([123, 227, 255])  # Lower bounds for color (in BGR format)
red_bounds_two = np.array([36, 29, 188])  # Upper bounds for color (in BGR format)

# Define the RGB range for leaf color
red_color_lower_bounds = np.minimum(red_bounds_one, red_bounds_two)
red_color_upper_bounds = np.maximum(red_bounds_one, red_bounds_two)

red_color_lower_bounds = np.array([19, 14, 49])
red_color_upper_bounds = np.array([51, 47, 237])

red_color_lower_bounds_one = np.array([15,50,106])
red_color_upper_bounds_two = np.array([218,238,255])


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


for img_path in tqdm.tqdm(image_paths, desc='Processing'):
    _, apple_bboxes = get_apple_info(img_path)
    total_apple_num, _ = get_apple_info(img_path)

    sample = cv2.imread(img_path)

    colour = sample.copy()

    apple_mask_1 = cv2.inRange(sample, red_color_lower_bounds, red_color_upper_bounds)
    apple_mask_2 = cv2.inRange(sample, red_color_lower_bounds_one, red_color_upper_bounds_two)

    final_mask = cv2.bitwise_or(apple_mask_1, apple_mask_1)

    result = cv2.bitwise_and(sample, sample, mask=final_mask)

    gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=33, sigmaY=33)

    divide = cv2.divide(gray_image, blur, scale=255)

    divide = np.invert(divide)

    kernel = np.ones((3, 2), np.uint8)

    closing = cv2.morphologyEx(divide, cv2.MORPH_CLOSE, kernel, iterations=2)

    radius = 4

    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    erosion = cv2.erode(closing, kernal, iterations=2)

    erosion = cv2.erode(erosion, kernal, iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations=1)

    inverted_image = cv2.bitwise_not(closing)
    labels, nlabels = ndimage.label(inverted_image)

    circles = cv2.HoughCircles(inverted_image, cv2.HOUGH_GRADIENT, 2, 20,
                               param1=30, param2=25,
                               minRadius=5, maxRadius=80)

    apple_coordinates = []

    if circles is not None:
        num_circles = circles.shape[1]  # 获取检测到的圆的数量
        # print(f"Number of circles: {num_circles}")
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            x, y = i[0], i[1]
            center = (x, y)
            apple_coordinates.append((x, y))
            # circle center
            cv2.circle(colour, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(colour, center, radius, (255, 0, 255), 3)


    # Close the existing OpenCV window

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for apple_coord in apple_coordinates:
        x, y = apple_coord

        for xmin, ymin, xmax, ymax in apple_bboxes:
            cv2.rectangle(colour, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2, cv2.LINE_AA)

        is_inside = any(xmin <= x <= xmax and ymin <= y <= ymax for xmin, ymin, xmax, ymax in apple_bboxes)

        # is_inside = any(xmin <= x <= xmax and ymin <= y <= ymax
        #                 for xmin, ymin, xmax, ymax in apple_bboxes)

        if is_inside:
            true_positive += 1
        else:
            false_positive += 1



    false_negative = len(apple_bboxes) - true_positive

    # precision = true_positive / (true_positive + false_positive)
    denominator = true_positive + false_positive
    precision = true_positive / denominator if denominator != 0 else 0
    accuracy = true_positive / (true_positive + false_positive + false_negative)

    # 打印结果
    # print(f"Precision: {precision}")
    # print(f"Accuracy: {accuracy}")
    # print(f"{img_path}")

    total_true_positive += true_positive
    total_false_positive += false_positive
    total_false_negative += false_negative

    # plt.figure(figsize=(12, 6))
    #
    # plt.subplot(1, 2, 1)
    # plt.imshow(cv2.cvtColor(colour, cv2.COLOR_BGR2RGB))
    # plt.title(f"Original Image {img_path}")
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(cv2.cvtColor(inverted_image, cv2.COLOR_BGR2RGB))
    # plt.title(f"Color Threshold Image {img_path}")
    # plt.show()


print(f"Length of the dataset: {len(image_paths)}")
overall_precision = total_true_positive / (total_true_positive + total_false_positive)
overall_accuracy = total_true_positive / (total_true_positive + total_false_positive + total_false_negative)
overall_recall = total_true_positive / (total_true_positive + total_false_negative)
# Print the overall results
print(f"Overall Precision: {overall_precision}")
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Recall: {overall_recall}")