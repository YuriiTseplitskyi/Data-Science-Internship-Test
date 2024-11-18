import cv2 
import matplotlib.pyplot as plt
import os
from typing import List, Tuple

def visualize_images(img1_path: str, img2_path: str):
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    _, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    ax[0].imshow(img1, cmap='gray' if len(img1.shape) == 2 else None)
    ax[0].set_title("Image 1")
    ax[0].axis("off")

    ax[1].imshow(img2, cmap='gray' if len(img2.shape) == 2 else None)
    ax[1].set_title("Image 2")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
    

def get_data(root_dir: str) -> List[Tuple[str, str]]:
    """
    In the given directory, finds all subdirectories containing pair of images.
    """
    images = []

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name)
        if os.path.isdir(folder_path): 
            file_paths = []
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".jpg"): 
                    full_path = os.path.join(folder_path, file_name)
                    file_paths.append(full_path)
            if file_paths:
                images.append(tuple(file_paths))

    return images