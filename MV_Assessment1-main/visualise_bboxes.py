# Import necessary libraries
import os
from utils.apple_dataset import AppleDataset
import cv2
from tqdm import trange

# Specify the path to the Minneapple Detection Dataset
image_dir = "/Users/guochaojun/Desktop/machine vision/detection/train"

try:
    # Create an instance of the AppleDataset class, providing the dataset root directory and no transformations
    app = AppleDataset(root_dir=image_dir, transforms=None)

    print(f"Showing images from {image_dir}")
    # Iterate over each image in the dataset
    for idx in trange(len(app)):

        # Obtain the name of the current image
        img_name = app.get_img_name(idx)

        # Construct the full path to the image using the dataset root directory
        img_path = os.path.join(image_dir, "images", img_name)
        
        # Check if the image file exists
        if not os.path.exists(img_path):
            print(f"[ ERROR ] Image file not found: {img_path}")
            continue

        # Access the bounding box information for the current image
        img_boxes = app[idx][1]['boxes']
        
        try:
            # Read the image using OpenCV
            img = cv2.imread(img_path)

            # Draw rectangles around the detected objects based on the bounding box coordinates
            for box in img_boxes:
                xmin = int(box[0].item())
                ymin = int(box[1].item())
                xmax = int(box[2].item())
                ymax = int(box[3].item())
                cv2.rectangle(img, (xmin, ymin),
                              (xmax, ymax), (0, 0, 255), 2, cv2.LINE_AA)

            # Display the image with bounding boxes
            cv2.imshow(f"Minneapple {idx+1}/{len(app)}", img)

            print(f"{len(img_boxes)} apples in {img_name}")
            
            # Wait for a key event and obtain the pressed key
            key = cv2.waitKey(0)

            # Check if the pressed key is 'q' or 'Q' to exit the application
            if key == ord('q') or key == ord('Q'):
                print('[ NOTE ] Exiting application')
                break

            # Close the existing OpenCV window
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"Error processing image {img_name}: {str(e)}")
            continue

finally:
    # Close all OpenCV windows in case of an exception or normal exit
    cv2.destroyAllWindows()
