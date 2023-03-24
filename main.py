import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import cv2

IMG_HEIGHT, IMG_WIDTH = 100, 100

for dirname, _, filenames in os.walk('D:/GitHubProjects/traffic_signs_classification/dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


def create_dataset(img_folder):
    img_data_array = []
    class_name = []

    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder, dir1)):
            image_path = os.path.join(img_folder, dir1, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name


# Create train and test datasets
train_images, train_labels = create_dataset(r'D:/GitHubProjects/traffic_signs_classification/dataset/test')
test_images, test_labels = create_dataset(r'D:/GitHubProjects/traffic_signs_classification/train')

print(train_labels)