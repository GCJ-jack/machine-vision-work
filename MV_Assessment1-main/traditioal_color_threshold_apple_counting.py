import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
from scipy import ndimage
from tqdm import trange


from utils.apple_dataset import AppleDataset

image_dir = "/Users/guochaojun/Desktop/MV_Assessment1_dataset/train"
# dataset = AppleDataset(root_dir=image_dir, transforms=None)

total_true_positive = 0
total_false_positive = 0
total_false_negative = 0

folder_path = "/Users/guochaojun/Desktop/MV_Assessment1_dataset/train"

image_paths = glob.glob(os.path.join(folder_path, "*.png"))

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




# def get_apple_num(image_path: str):
#     xml_path = image_path.replace(".png", ".xml")
#     assert os.path.isfile(xml_path), f"File not found: {xml_path}"
#
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#
#     apple_count = 0
#
#     for obj_elem in root.findall('.//object'):
#         name_elem = obj_elem.find('name')
#
#         if name_elem is not None and name_elem.text == 'apple':
#             apple_count += 1
#
#     return apple_count
#
#
# for image_path in image_paths:
#     # print(image_path)
#
#     print(f"{image_path}: {get_apple_num(image_path)} apples")
#
#     xml_path = image_path.replace(".png", ".xml")
    # print(xml_path)

green_bounds_one = np.array([1, 219, 220])  # Lower bounds for color (in BGR format)
green_bounds_two = np.array([226, 248, 255])  # Upper bounds for color (in BGR format)

green_bounds_one = np.array([59, 153, 141])  # Lower bounds for color (in BGR format)
green_bounds_two = np.array([188, 254, 255])  # Upper bounds for color (in BGR format)

green_lower_bound = np.minimum(green_bounds_one, green_bounds_two)
green_upper_bound = np.maximum(green_bounds_one, green_bounds_two)

red_bounds_one = np.array([123, 227, 255])  # Lower bounds for color (in BGR format)
red_bounds_two = np.array([36, 29, 188])  # Upper bounds for color (in BGR format)

# Define the RGB range for leaf color
red_color_lower_bounds = np.minimum(red_bounds_one, red_bounds_two)
red_color_upper_bounds = np.maximum(red_bounds_one, red_bounds_two)

red_color_lower_bounds = np.array([19, 14, 49])
red_color_upper_bounds = np.array([51, 47, 237])

dataset = glob(os.path.join(image_dir, '*.png'))

try:
    app = AppleDataset(root_dir=image_dir, transforms=None)
    for idx in trange(len(app)):

        img = dataset.get_img_name(idx)

        img_path = os.path.join(image_dir, "images", img)

        if not os.path.exists(img_path):
            print(f"[ ERROR ] Image file not found: {img_path}")
            continue

        sample = cv2.imread(img_path)

        colour = sample.copy()

        # sample = cv2.medianBlur(sample, 25)

        # make a deep copy of the original image

        # Exclude the leaf color range

        # block_size, c_value = calculate_block_size_and_c(sample)
        apple_mask_1 = cv2.inRange(sample, green_lower_bound, green_upper_bound)
        apple_mask_2 = cv2.inRange(sample, red_color_lower_bounds, red_color_upper_bounds)

        final_mask = cv2.bitwise_or(apple_mask_2, apple_mask_2)

        result = cv2.bitwise_and(sample, sample, mask=final_mask)

        kernel = np.ones((3, 3), np.uint8)

        gray_image = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=33, sigmaY=33)

        divide = cv2.divide(gray_image, blur, scale=255)

        divide = np.invert(divide)

        kernel = np.ones((3, 2), np.uint8)

        closing = cv2.morphologyEx(divide, cv2.MORPH_CLOSE, kernel, iterations=2)

        radius = 4

        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

        erosion = cv2.erode(closing, kernal, iterations=2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel, iterations=1)

        # kernel = np.ones((8, 8), np.uint8)
        # erosion2 = cv2.erode(closing, kernel, iterations=2)
        #
        # erosion2[:1, :] = 0
        # erosion2[:, :1] = 0
        # erosion2[-1:, :] = 0
        # erosion2[:, -1:] = 0
        #
        # radius = 3
        #
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        #
        # dilated_image = cv2.dilate(erosion2, kernel, iterations=2)
        #
        # opening = cv2.morphologyEx(dilated_image, cv2.MORPH_OPEN, kernel, 1)

        # radius = 3  # Adjust the radius according to your needs
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        #
        # closed_image = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

        inverted_image = cv2.bitwise_not(closing)
        labels, nlabels = ndimage.label(inverted_image)
        # print("There are " + str(nlabels) + " apples")

        # centroid = ndimage.center_of_mass(inverted_image, labels, np.arange(nlabels) + 1)

        apple_coordinates = []
        # for cen in centroid:
        #     colour = cv2.circle(colour, (cen[1].astype(int), cen[0].astype(int)), radius=15, color=(255, 255, 255),
        #                         thickness=2)
        #     y, x = cen[:2]  # Assuming the coordinates are (y, x) or (row, column)
        #     apple_coordinates.append((x, y))

        circles = cv2.HoughCircles(inverted_image, cv2.HOUGH_GRADIENT, 2, 20,
                                  param1=30, param2=25,
                                  minRadius=5, maxRadius=80)

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
                # radius = i[2]
                # cv2.circle(inverted_image, center, radius, (255, 0, 255), 3)

        img_boxes = app[idx][1]['boxes']
        # # Read the image using OpenCV
        img = cv2.imread(img_path)
        #
        img_name = app.get_img_name(idx)
        #
        mask_coordinates = []
        # Draw rectangles around the detected objects based on the bounding box coordinates
        for box in img_boxes:
            xmin = int(box[0].item())
            ymin = int(box[1].item())
            xmax = int(box[2].item())
            ymax = int(box[3].item())
            cv2.rectangle(img, (xmin, ymin),
                          (xmax, ymax), (0, 0, 255), 2, cv2.LINE_AA)
            mask_coordinates.append((xmin, ymin, xmax, ymax))

        # Display the image with bounding boxes using OpenCV
        cv2.imshow(f"Minneapple {idx + 1}/{len(app)}", img)
        #
        print(f"{len(img_boxes)} apples in {img_name}")
        #
        # Wait for a key event and obtain the pressed key
        key = cv2.waitKey(0)

        # Check if the pressed key is 'q' or 'Q' to exit the application
        if key == ord('q') or key == ord('Q'):
            print('[ NOTE ] Exiting application')

        # Close the existing OpenCV window
        cv2.destroyAllWindows()
        #
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(colour, cv2.COLOR_BGR2RGB))
        plt.title(f"Original Image {img_name}")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(inverted_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Color Threshold Image {img_name}")

        plt.show()

        true_positive = 0
        false_positive = 0
        false_negative = 0

        for apple_coord in apple_coordinates:
            x, y = apple_coord
            is_inside = any(xmin <= x <= xmax and ymin <= y <= ymax
                            for xmin, ymin, xmax, ymax in mask_coordinates)

            if is_inside:
                true_positive += 1
            else:
                false_positive += 1
        false_negative = len(mask_coordinates) - true_positive

        # precision = true_positive / (true_positive + false_positive)
        denominator = true_positive + false_positive
        precision = true_positive / denominator if denominator != 0 else 0
        accuracy = true_positive / (true_positive + false_positive + false_negative)

        # 打印结果
        print(f"Precision: {precision}")
        print(f"Accuracy: {accuracy}")
        print(f"{img_path}")

        total_true_positive += true_positive
        total_false_positive += false_positive
        total_false_negative += false_negative
finally:
    cv2.destroyAllWindows()
print(f"Length of the dataset: {len(dataset)}")
overall_precision = total_true_positive / (total_true_positive + total_false_positive)
overall_accuracy = total_true_positive / (total_true_positive + total_false_positive + total_false_negative)
overall_recall = total_true_positive / (total_true_positive + total_false_negative)
# Print the overall results
print(f"Overall Precision: {overall_precision}")
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Recall: {overall_recall}")
