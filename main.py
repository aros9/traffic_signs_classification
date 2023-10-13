"""
Traffic Signs Classification based on Convolutional Neural Network.
"""

import tensorflow as tf
import os

from support import (
    create_dataset,
    extract_image_and_label,
    create_test_and_validation_dataset,
    show_images
)

from aug_image import augment_images

# Read file names from directory
list_dataset = tf.data.Dataset.list_files(os.getcwd() + r'/dataset/*/*', shuffle=False)

# Create labeled dataset from list of files
labeled_dataset = list_dataset.map(extract_image_and_label)

# Perform image augmentation on given dataset
labeled_dataset = labeled_dataset.map(augment_images)

show_images(labeled_dataset, num_images=2)

