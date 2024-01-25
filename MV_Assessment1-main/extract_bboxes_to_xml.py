# Import necessary libraries
import os
from utils.apple_dataset import AppleDataset
import cv2
from tqdm import trange
from pascal_voc_writer import Writer

# Specify the path to the Minneapple Detection Dataset
root_dir = "/Users/guochaojun/Desktop/machine vision/detection/train"

# Specify the directory to save the XML labels
xml_label_dir = "/Users/guochaojun/Desktop/machine vision/detection/labels/xml"

# Create the output directory if it doesn't exist
os.makedirs(xml_label_dir, exist_ok=True)

# Create an instance of the AppleDataset class, providing the dataset root directory and no transformations
app = AppleDataset(root_dir=root_dir, transforms=None)

# Print a message indicating the start of XML label generation
print(f"Generating XML labels from {root_dir}")

# Iterate over each image in the dataset using tqdm for a progress bar
for idx in trange(len(app), desc="Processing"):

    # Obtain the name of the current image
    img_name = app.get_img_name(idx)

    # Construct the full path to the image using the dataset root directory
    img_path = os.path.join(root_dir, "images", img_name)

    # Specify the path to save the XML label file
    xml_label_path = os.path.join(xml_label_dir, f"{os.path.splitext(img_name)[0]}.xml")

    # Check if the image file exists
    if not os.path.exists(img_path):
        print(f"[ ERROR ] Image file not found: {img_path}")
        continue

    # Access the bounding box information for the current image
    img_boxes = app[idx][1]['boxes']

    try:
        # Read the image using OpenCV
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape

        # Create an instance of the PascalVoc Writer for the current image
        writer = Writer(img_path, img_w, img_h)

        # Draw rectangles around the detected objects based on the bounding box coordinates
        for box in img_boxes:
            xmin = int(box[0].item())
            ymin = int(box[1].item())
            xmax = int(box[2].item())
            ymax = int(box[3].item())

            # Add each object (e.g., "apple") with its bounding box coordinates to the XML file
            writer.addObject("apple", xmin, ymin, xmax, ymax)
        
        # Save the XML label file for the current image
        writer.save(xml_label_path)

    except Exception as e:
        print(f"Error processing image {img_name}: {str(e)}")
        continue
