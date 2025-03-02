import os
import shutil
import subprocess
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#helps to control a subset of the dataset for sanity checks
# TODO:delete this file when no longer needed 


def move_directories(source_dir, dest_dir):
 
    # List all directories in source_dir, sorted alphabetically
    try:
        directories = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        directories.sort()  # Sort directories alphabetically
    except FileNotFoundError:
        print(f"Error: Source directory '{source_dir}' does not exist.")
        return
    except PermissionError:
        print(f"Error: Permission denied to access '{source_dir}'.")
        return

    # Check if we have more than 100 directories to move
    if len(directories) <= 100:
        print(f"There are {len(directories)} directories in '{source_dir}', no directories will be moved.")
        return

    # Directories to move (everything after the 100th one)
    directories_to_move = directories[100:]

    # Move directories to the destination
    for dir_name in directories_to_move:
        source_path = os.path.join(source_dir, dir_name)
        dest_path = os.path.join(dest_dir, dir_name)

        try:
            shutil.move(source_path, dest_path)
            print(f"Moved: {dir_name}")
        except Exception as e:
            print(f"Error moving '{dir_name}': {e}")

    print(f"Completed moving {len(directories_to_move)} directories to '{dest_dir}'.")


def get_sorted_checkpoints(directory):
    # List to hold the relative paths and their associated numeric values
    checkpoints = []

    # Walk through the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file matches the pattern 'checkpoint_*.pth'
            match = re.match(r'checkpoint_(\d+)\.pth', file)
            if match:
                # Extract the number from the filename
                number = int(match.group(1))
                # Get the relative path of the file
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                # Append tuple (number, relative_path)
                checkpoints.append((number, relative_path))

    # Sort the checkpoints by the number
    checkpoints.sort(key=lambda x: x[0])

    # Return just the sorted relative paths
    return [f'{directory}/{relative_path}'  for _, relative_path in checkpoints]


def create_dirs(path):
    os.makedirs(path,exist_ok=True)


def create_image(path,i):
    tensor_data = np.load(path)  # Shape: [1, 3, 224, 224]
    print(tensor_data.shape)
    tensor_data = tensor_data.squeeze(0)  # Now it should be [3, 224, 224]

    image_data = np.transpose(tensor_data, (1, 2, 0))  # Now it's [224, 224, 3]

    image_data = (image_data * 255).astype(np.uint8)
    image = Image.fromarray(image_data)
    # Create a PIL image
    #image = Image.fromarray(image_data)
    plt.imsave(f"testing/pert_vis/image_{i}.png",image_data)
  

# Example usage:
source_directory = 'val'
destination_directory = 'train'

#move_directories(source_directory, destination_directory)

#x = get_sorted_checkpoints("finetuned_models/batchnorm/")
#print(x)

for i in range(10):
    create_image(f'testing/pert_vis/745/pert_{i}.npy',i)