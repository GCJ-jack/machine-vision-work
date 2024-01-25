# Import necessary libraries
import os  # Import the os library for file and directory operations
import numpy as np  # Import NumPy for numerical operations
from utils.apple_dataset import AppleDataset  # Import the custom AppleDataset class
import cv2  # Import OpenCV for image processing
from tqdm import trange  # Import tqdm for creating progress bars

# Specify the path to the Minneapple Detection Dataset
root_dir = "/Users/guochaojun/Desktop/machine vision/detection/train"

# Specify the directory to save the XML labels
yolo_label_dir = "machine Vision/Assessment 1/MV_Assessment1/datasets/Minneapple/yolo/"

# Create the output directory if it doesn't exist
os.makedirs(yolo_label_dir, exist_ok=True)

# Create an instance of the AppleDataset class, providing the dataset root directory and no transformations
app = AppleDataset(root_dir=root_dir, transforms=None)

# Print a message indicating the start of XML label generation
print(f"Generating YOLO labels from {root_dir}")

# Iterate over each image in the dataset using tqdm for a progress bar
for idx in trange(len(app), desc="Processing"):
    # Obtain the name of the current image
    img_name = app.get_img_name(idx)

    # Construct the full path to the image using the dataset root directory
    img_path = os.path.join(root_dir, "images", img_name)

    # Specify the path to save the XML label file
    yolo_label_path = os.path.join(yolo_label_dir, f"{os.path.splitext(img_name)[0]}.txt")

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

        with open(yolo_label_path, "w") as f:
            # Draw rectangles around the detected objects based on the bounding box coordinates
            for box in img_boxes:
                xmin = int(box[0].item())
                ymin = int(box[1].item())
                xmax = int(box[2].item())
                ymax = int(box[3].item())

                # Check for invalid boxes
                if xmin == xmax or ymin == ymax:
                    print(f" [ ERROR ] Found invalid box with coords: [{xmin}, {ymin}, {xmax}, {ymax}]")
                    continue

                # Calculate box center coordinates and dimensions
                cx = (xmax + xmin) // 2
                cy = (ymax + ymin) // 2
                w = xmax - xmin
                h = ymax - ymin

                # Normalize coordinates and dimensions
                cx_norm = np.round(cx / img_w, decimals=4)
                cy_norm = np.round(cy / img_h, decimals=4)
                w_norm = np.round(w / img_w, decimals=4)
                h_norm = np.round(h / img_h, decimals=4)

                # Write YOLO format label to file
                f.write(f"{0} {cx_norm} {cy_norm} {w_norm} {h_norm}\n")
        
    except Exception as e:
        print(f"Error processing image {img_name}: {str(e)}")
        continue
