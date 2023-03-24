import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cv2

from create_dataset import create_dataset

for dirname, _, filenames in os.walk('D:/GitHubProjects/traffic_signs_classification/dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Create train and test datasets
train_images, train_labels = create_dataset(r'D:/GitHubProjects/traffic_signs_classification/dataset/test')
test_images, test_labels = create_dataset(r'D:/GitHubProjects/traffic_signs_classification/train')

print(train_labels)