"""
File contains functions related to image augmentation used in project.
Note that running this script will increase size of dataset stored in
dataset directory.
"""

import tensorflow as tf
import os
import math

from PIL import Image

ROT_ANGLE = 10

INPUT_DIR = os.getcwd() + r'\dataset'
OUTPUT_DIR = os.getcwd() + r'\dataset_augmented'

# Tensorflow model used for data augmentation
aug_model = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(factor=ROT_ANGLE * math.pi / 180),
    # tf.keras.layers.RandomBrightness(factor=0.001, value_range=[-0.1, 0.1]),
    tf.keras.layers.RandomContrast(factor=0.1),
    tf.keras.layers.GaussianNoise(stddev=0.01)
])


def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    """
    Helper function for applying gaussian blur.

    :param kernel_size:
    :param sigma:
    :param n_channels:
    :param dtype:

    :return:
    """
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def apply_gaussian_blur(img):
    """
    Function that applies gaussian blur to image.

    :param img: Image that will be blured.

    :return img: Blurred image.
    """
    blur = _gaussian_kernel(3, 2, 1, img.dtype)
    img = tf.nn.depthwise_conv2d(img[None], blur, [1, 1, 1, 1], 'SAME')
    return img[0]


def augment_images(img, label):
    """
    Function that does image augmentation on a given dataset. This functions covers only a few methods, such as
    randomly changing brightness, contrast, rotating by up to 10 degrees and applying gaussian blur.

    :param img: Image in tensorflow format.
    :param label: Label corresponding to the image.

    :return img, label: Dataset that contains images and corresponding labels.
    """

    image = aug_model(img)

    return image, label


def create_augmented_dataset(input_dir: str, output_dir: str):
    """
    Function expands already existing dataset from input_dir and saves augmented files into the output directory.

    :param input_dir: Path to input directory.
    :param output_dir: Path to output directory.
    """

    # Get the list of image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg') or f.endswith('.png')]

    # Loop through each image file and perform augmentation
    for image_file in image_files:
        # Load the image
        img = Image.open(os.path.join(input_dir, image_file))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)  # Reshape to (1, height, width, channels)

        # Generate augmented images
        i = 0
        for _ in range(20):  # Generate 20 augmented images per input image
            augmented_img_array = aug_model(img_array)

            # Save augmented image
            augmented_img = tf.keras.preprocessing.image.array_to_img(augmented_img_array[0])
            augmented_img.save(os.path.join(output_dir, f'aug_{i}_{image_file}'))
            i += 1


if __name__ == "__main__":

    # Perform data augmentation and save augmented images
    for sub_dir in os.listdir(INPUT_DIR):
        input_path = os.path.join(INPUT_DIR, sub_dir)
        output_path = os.path.join(OUTPUT_DIR, sub_dir)
        create_augmented_dataset(input_path, output_path)
