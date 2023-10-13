"""
File contains supporting functions to project: traffic_signs_classification
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

IMG_HEIGHT, IMG_WIDTH = 64, 64
BATCH_SIZE = 2


def create_dataset(img_folder_path: str) -> tuple[list, list]:
    """
    Function that takes image folder path and reads all the data into numpy array.

    :param img_folder_path: Path to the folder containing images.

    :return: Tuple containing: img_data_array - array that contains read image files,
                              class_name - array that contains read image class names.
    """

    img_data_array, class_name = [], []

    for dir1 in os.listdir(img_folder_path):
        for file in os.listdir(os.path.join(img_folder_path, dir1)):

            # Extract single file path from directory
            image_path = os.path.join(img_folder_path, dir1, file)

            # Read image
            image = cv2.imread(image_path, cv2.COLOR_RGB2GRAY)

            # Resize image to proper height and width
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)

            # Convert image into numpy array and scale it to 0 ... 1
            image = np.array(image)
            image = image.astype('float32')
            image /= 255

            # Append image and label into list
            img_data_array.append(image)
            class_name.append(dir1)

    # Convert class names to integer numbers
    target_dict = {k: v for v, k in enumerate(np.unique(class_name))}
    class_name = [target_dict[class_name[i]] for i in range(len(class_name))]

    return img_data_array, class_name


def extract_image_and_label(file_path: str):
    """
    Function processes dataset on given file path and extract image files.

    :param file_path: Path to dataset directory

    """

    # Extract label from file path
    label = tf.strings.split(file_path, os.path.sep)[-2]

    # Read the image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)

    # Convert RGB to Grayscale
    image = tf.image.rgb_to_grayscale(image)

    # Set the explicit shape
    image.set_shape([None, None, 1])

    # Resize the image
    image = tf.image.resize(image, (IMG_HEIGHT, IMG_WIDTH))
    image = image/255

    return image, label


def create_test_and_validation_dataset(dir_path: str):
    """
    Function creates two subsets from given directory. First one is training set that
    takes 90% of directory data and second one is validation subset that contains remaining
    10% of the data.

    :param dir_path: Path to directory with subfolders.

    :return:
    """
    ds_train = tf.keras.preprocessing.image_dataset_from_directory(
        directory=dir_path,
        labels='inferred',
        label_mode="int",
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset="training"
    )

    ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
        directory=dir_path,
        labels='inferred',
        label_mode="int",
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        shuffle=True,
        seed=123,
        validation_split=0.1,
        subset="validation"
    )

    return ds_train, ds_validation


def show_images(dataset, num_images=1):
    """
    Function that displays given number of images from tf dataset.

    :param dataset: Dataset that contains array of images and labels
    :param num_images: Number of displayed images.
    """
    for image, l in dataset.take(num_images):
        print("L = ", l)
        plt.figure()
        plt.imshow(tf.squeeze(image.numpy(), axis=-1), cmap='gray')
        plt.axis('off')
    plt.show()
