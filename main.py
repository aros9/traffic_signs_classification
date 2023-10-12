"""
Traffic Signs Classification based on Convolutional Neural Network.
"""

import os
from support import create_dataset

images, labels = create_dataset(os.getcwd() + '/dataset')
