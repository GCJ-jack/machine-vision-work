import os
import cv2
import numpy as np
import imutils
import xml.etree.ElementTree as ET


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
    total_apple_num, _ = get_apple_info(img_path)

    original_img = cv2.imread(img_path)

    img = original_img.copy()

    img_h, img_w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    sat_lower = 25
    val_lower = 25

    mask1 = cv2.inRange(hsv, (0, sat_lower, val_lower), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, sat_lower, val_lower), (180, 255, 255))

    mask = cv2.bitwise_or(mask1, mask2)

    # kernel = np.ones((5, 5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # print(kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    min_contour_area = 200
    filtered_contours = [
        cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    result_img = img.copy()
    cv2.drawContours(result_img, filtered_contours, -1, (0, 255, 0), 2)

    for apple_bbox in apple_bboxes:
        cv2.rectangle(
            result_img, apple_bbox[:2], apple_bbox[2:4], (12, 192, 255), 2, cv2.LINE_AA)

    circles = cv2.HoughCircles(
        mask,
        method=cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=12.5,  # smaller values help detecting cluttered circles
        param1=100,  # higher threshold of Canny edge detector - 300 for cv2.HOUGH_GRADIENT_ALT
        param2=8,  # cv2.HOUGH_GRADIENT - smaller means more false circles | cv2.HOUGH_GRADIENT_ALT - circle perfectness measure
        minRadius=7,
        maxRadius=50)

    if circles is None:
        print("Found 0 circles")
    else:
        print(f"Found {len(np.array(circles).reshape(-1, 3))} circles")

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(mask, (x, y), r, (0, 0, 255), 4)
            # cv2.rectangle(mask, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    cnt_centres = set()

    for contour in filtered_contours:
        for bbox_count, bbox in enumerate(apple_bboxes):
            in_box = False

            contour = contour.reshape(-1, 2)

            xmin, ymin, xmax, ymax = bbox

            x_cnt = int(np.mean(contour[:, 0]))
            y_cnt = int(np.mean(contour[:, 1]))

            cnt_centres.add((x_cnt, y_cnt))

            if (ymin <= y_cnt <= ymax) and (xmin <= x_cnt <= xmax):

                cv2.circle(img, (x_cnt, y_cnt),
                           5, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.rectangle(img, (xmin, ymin),
                              (xmax, ymax), (12, 192, 255), 2, cv2.LINE_AA)

                in_box = True

                true_positives += 1

                break

            if not in_box and bbox_count == (len(apple_bboxes) - 1):
                cv2.circle(img, (x_cnt, y_cnt),
                           5, (0, 0, 255), -1, cv2.LINE_AA)

                false_positives += 1

    for xmin, ymin, xmax, ymax in apple_bboxes:
        for idx, cnt in enumerate(cnt_centres):
            not_in = True
            if (xmin <= cnt[0] <= xmax) and (ymin <= cnt[1] <= ymax):
                not_in = False
                break
            if not_in and (idx == len(cnt_centres) - 1):

                cv2.rectangle(img, (xmin, ymin),
                              (xmax, ymax), (255, 0, 0), 2, cv2.LINE_AA)

                false_negatives += 1

    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)
    f1_score = 2 * ((precision * recall) / (precision + recall))
    
    true_positives_circle = 0
    false_positives_circle = 0
    false_negatives_circle = 0

    circle_centres = set()

    for circle in circles:
        for bbox_count, bbox in enumerate(apple_bboxes):
            in_box = False

            xmin, ymin, xmax, ymax = bbox

            x_circle = circle[0]
            y_circle = circle[1]

            circle_centres.add((x_circle, y_circle))

            if (ymin <= y_circle <= ymax) and (xmin <= x_circle <= xmax):

                cv2.circle(img, (x_circle, y_circle),
                           5, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.rectangle(img, (xmin, ymin),
                              (xmax, ymax), (12, 192, 255), 2, cv2.LINE_AA)

                in_box = True

                true_positives_circle += 1

                break

            if not in_box and bbox_count == (len(apple_bboxes) - 1):
                cv2.circle(img, (x_circle, y_circle),
                           5, (0, 0, 255), -1, cv2.LINE_AA)

                false_positives_circle += 1

    for xmin, ymin, xmax, ymax in apple_bboxes:
        for idx, crlc in enumerate(circle_centres):
            not_in = True
            if (xmin <= crlc[0] <= xmax) and (ymin <= crlc[1] <= ymax):
                not_in = False
                break
            if not_in and (idx == len(cnt_centres) - 1):

                cv2.rectangle(img, (xmin, ymin),
                              (xmax, ymax), (255, 0, 0), 2, cv2.LINE_AA)

                false_negatives_circle += 1

    precision_circle = true_positives_circle/(true_positives_circle + false_positives_circle)
    recall_circle = true_positives_circle/(true_positives_circle + false_negatives_circle)
    f1_score_circle = 2 * ((precision_circle * recall_circle) / (precision_circle + recall_circle))

    display_images(original_img, img, result_img, mask)

    return (precision, recall, f1_score), (precision_circle, recall_circle, f1_score_circle)


def display_images(og_img, img_, rslt_img, mask):
    img_h, img_w = img_.shape[:2]

    separator = np.ones((img_h, 10, 3), np.uint8) * 128

    display_image = cv2.hconcat(
        [og_img, separator, img_, separator, rslt_img, separator, mask])
    display_image = imutils.resize(display_image, height=840)
    cv2.imshow("img", display_image)
    
    cv2.imwrite('/tmp/output.png', display_image)

    # cv2.imshow("rslt_img", rslt_img)

    # cv2.imshow("mask", imutils.resize(mask, height=840))

    cv2.waitKey(0)

    cv2.destroyAllWindows()


root_dir = "/home/dexter/Uni/UWE/Machine Vision/Assessment 1/Datasets/Minneapple/detection/train/images/Red/test/"
img_path = os.path.join(root_dir, "20150921_131346_image21.png")

_, apple_bboxes = get_apple_info(img_path)
contour_scores, circle_scores = identify_apples_evaluate_detection(
    img_path, apple_bboxes)

precision, recall, f1_score = contour_scores
precision_circle, recall_circle, f1_score_circle = circle_scores

print("Contour Scores")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f} \n")

print("Circle Scores")
print(f"Precision: {precision_circle:.4f}")
print(f"Recall: {recall_circle:.4f}")
print(f"F1-Score: {f1_score_circle:.4f}")


# pos_x = [0 8.7357 9.7405 -8.7357 -9.7405];
# pos_y = [29.0587 32.6035 36.3537 32.6035 36.3537];
# pos_z = [36.1785 31.7238 15.0006 31.7238 15.0006];
