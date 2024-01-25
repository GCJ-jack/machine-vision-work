# Import necessary libraries
import os
import glob
import cv2
import xml.etree.ElementTree as ET
import tqdm  # A library for creating progress bars

# Specify the path to the Minneapple Detection Dataset
image_dir = "/home/dexter/Uni/UWE/Machine Vision/Assessment 1/Datasets/Minneapple/detection/train"
xml_label_dir = "/home/dexter/Uni/UWE/Machine Vision/Assessment 1/Datasets/Minneapple/detection/labels/xml"

# Get a list of all XML files in the specified directory
xml_files = glob.glob(os.path.join(xml_label_dir, "*.xml"))

# Iterate over each XML file
# tqdm is used to create a progress bar for iteration
for xml_file_path in tqdm.tqdm(xml_files):

    # Obtain the name of the current XML file
    xml_name = os.path.splitext(os.path.basename(xml_file_path))[0]

    # Construct the corresponding image file name
    img_name = f"{xml_name}.png"

    # Construct the full path to the image using the dataset root directory
    img_path = os.path.join(image_dir, "images", img_name)

    # Check if the image file exists
    if not os.path.exists(img_path):
        print(f"[ ERROR ] Image file not found: {img_path}")
        continue  # Skip to the next iteration if the image file is not found

    # Read the image using OpenCV
    img = cv2.imread(img_path)

    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Iterate over each object in the XML file
    for obj in root.findall('.//object'):
        name = obj.find('name').text
        xmin = int(obj.find('.//xmin').text)
        ymin = int(obj.find('.//ymin').text)
        xmax = int(obj.find('.//xmax').text)
        ymax = int(obj.find('.//ymax').text)

        # Draw a rectangle around the object on the image
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax),
                      (0, 0, 255), 2, cv2.LINE_AA)

    # Display the image with bounding boxes
    cv2.imshow(f"{img_name}", img)

    # Wait for a key event and obtain the pressed key
    key = cv2.waitKey(0)

    # Check if the pressed key is 'q' or 'Q' to exit the application
    if key == ord('q') or key == ord('Q'):
        print('[ NOTE ] Exiting application')
        break

    # Close any open OpenCV window
    cv2.destroyAllWindows()

# Close any open OpenCV window
cv2.destroyAllWindows()
