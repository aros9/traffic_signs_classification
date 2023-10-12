"""
File contains supporting functions to project: traffic_signs_classification
"""

import os
import cv2
import numpy as np

IMG_HEIGHT, IMG_WIDTH = 64, 64


def create_dataset(img_folder_path: str) -> tuple[list, list]:
    """
    Function that takes image folder path and reads all the data into numpy array.

    param: img_folder_path: Path to the folder containing images.

    return: Tuple containing: img_data_array - array that contains read image files,
                              class_name - array that contains read image class names.
    """

    img_data_array, class_name = [], []

    for dir1 in os.listdir(img_folder_path):
        for file in os.listdir(os.path.join(img_folder_path, dir1)):
            image_path = os.path.join(img_folder_path, dir1, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)

    target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
    class_name = [target_dict[class_name[i]] for i in range(len(class_name))]
    return img_data_array, class_name
