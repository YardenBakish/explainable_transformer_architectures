import os
import shutil



def check(document1_path, parent_dir, target_dir):
    with open(document1_path, 'r') as file:
        subdirectories = [line.strip() for line in file]
    
    for subdirectory in subdirectories:
        target_subdir = os.path.join(target_dir, subdirectory)
         
        if os.path.isdir(target_subdir) == False:
            print(target_subdir)
    

def copy_subdirectories_from_file(document1_path, parent_dir, target_dir):
    """
    Reads a file (document1_path) containing subdirectory names, checks if
    they exist in the given parent directory, and copies their contents to the target directory.

    :param document1_path: Path to the document file containing subdirectory names.
    :param parent_dir: Path to the parent directory where the subdirectories are located.
    :param target_dir: Path to the target directory where contents will be copied to.
    """
    # Ensure the target directory exists

    
    # Open the document containing subdirectory names
    with open(document1_path, 'r') as file:
        subdirectories = [line.strip() for line in file]
    c =0
    for subdirectory in subdirectories:
        subdirectory = subdirectory.strip()  # Remove any surrounding whitespace/newlines

        # Construct the full path to the subdirectory
        subdir_path = os.path.join(parent_dir, subdirectory)
        target_subdir = os.path.join(target_dir, subdirectory)
        if os.path.isdir(subdir_path):
            shutil.copytree(subdir_path, target_subdir)
        else:
            print("f'{subdir_path} not exists'")
            exit(1)
        
        print(f'{c}/{len(subdirectories)}')
        c+=1
        # Check if the subdirectory exists
        
# Example usage
document1_path = '/home/ai_center/ai_users/yardenbakish/xai_guided_architecture/playground/tmp_labels.txt'
parent_dir = '/home/ai_center/ai_users/zimerman1/datasets/Imagenet/data/train'  # Path to the parent directory
target_dir = '/home/ai_center/ai_users/yardenbakish/xai_guided_architecture/train'  # Path to the target directory

copy_subdirectories_from_file(document1_path, parent_dir, target_dir)
# check(document1_path, parent_dir, target_dir)