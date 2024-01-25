import os
import shutil
from glob import glob

def separate_png_xml(source_folder, destination_folder_png, destination_folder_xml):
    # Ensure destination folders exist
    os.makedirs(destination_folder_png, exist_ok=True)
    os.makedirs(destination_folder_xml, exist_ok=True)

    # Get a list of all files in the source folder with extension .png or .xml
    files = glob(os.path.join(source_folder, '*.[px][nml]*'))

    # Separate files based on extension
    for file in files:
        filename, extension = os.path.splitext(file)

        if extension.lower() == '.png':
            shutil.move(file, os.path.join(destination_folder_png, os.path.basename(file)))
        elif extension.lower() == '.xml':
            shutil.move(file, os.path.join(destination_folder_xml, os.path.basename(file)))

# Example usage
source_directory = '/Users/guochaojun/Desktop/MV_Assessment1_dataset/train'
destination_directory_png = '/Users/guochaojun/Desktop/MV_Assessment1_dataset/img'
destination_directory_xml = '/Users/guochaojun/Desktop/MV_Assessment1_dataset/xml'

separate_png_xml(source_directory, destination_directory_png, destination_directory_xml)
