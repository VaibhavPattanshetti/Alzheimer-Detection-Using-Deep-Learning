import os
import imghdr
from PIL import Image
import numpy as np

def remove_invalid_images(data_path, valid_types=("jpeg", "png", "jpg")):
    removed = 0
    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                file_type = imghdr.what(file_path)
                if file_type not in valid_types:
                    os.remove(file_path)
                    removed += 1
            except:
                os.remove(file_path)
                removed += 1
    print(f"Total removed files: {removed}")


def check_image_channels(image_path):
    img = Image.open(image_path)
    arr = np.array(img)
    return arr.shape
