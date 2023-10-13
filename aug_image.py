"""
File contains functions related to image augmentation used in project.
"""

import tensorflow as tf
# import keras_cv
import math

ROT_ANGLE = 10

# Tensorflow model used for data augmentation
aug_model = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(factor=ROT_ANGLE * math.pi / 180),
    tf.keras.layers.RandomBrightness(factor=0.01, value_range=[0, 1]),
    tf.keras.layers.RandomContrast(factor=0.01),
    tf.keras.layers.GaussianNoise(stddev=0.6)
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
