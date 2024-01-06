import tensorflow as tf
from keras import layers, models
import os
from support import extract_image_and_label_detection
from PIL import Image
import numpy as np
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

BACKGROUND_IMG_SIZE = (200, 200, 1)

# # Read file names from directory
# list_dataset = tf.data.Dataset.list_files(os.getcwd() + r'/dataset_augmented/*/*', shuffle=True)
# list_validation = tf.data.Dataset.list_files(os.getcwd() + r'/dataset/*/*', shuffle=True)
#
# # Create labeled dataset from list of files
# labeled_dataset = list_dataset.map(extract_image_and_label_detection)
# validation_dataset = list_validation.map(extract_image_and_label_detection)
#
#
# print(f"Dataset length: {len(labeled_dataset)}\n")
#
# train_size = int(0.7 * len(labeled_dataset))
#
# ds_train = labeled_dataset.take(train_size)
# ds_test = labeled_dataset.skip(train_size)
#
# batch_size = 64
# ds_train = ds_train.batch(batch_size)
# ds_test = ds_test.batch(batch_size)

""" TEST CHANGES"""
# Initialize lists to store data
images = []
coordinates = []
dataset_path = os.getcwd() + '/dataset_detection/dataset'

# Load and preprocess images
for filename in os.listdir(dataset_path):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(dataset_path, filename)).convert('L').resize((BACKGROUND_IMG_SIZE[0], BACKGROUND_IMG_SIZE[1]))
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=-1)
        images.append(img_array)

        # Extract coordinates from the filename (assuming filename 'x_(x-position)y_(y-position).png'
        match = re.match(r"x(\d+)_y(\d+)\.png", filename)
        if match:
            x, y = map(int, match.groups())
            coordinates.append([x, y, x + 64, y + 64])

# Convert lists to NumPy arrays
X = np.array(images)
y = np.array(coordinates)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Optionally, you can shuffle and batch the datasets
batch_size = 32
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

"""END OF TEST CHANGES"""
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=BACKGROUND_IMG_SIZE))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten layer
model.add(layers.Flatten())

# Dense layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(4, activation='linear'))  # 4 output nodes for (x1, y1, x2, y2)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train model
history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=test_dataset,
)

# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.plot(history.history['loss'], label='loss')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# plt.show()

plt.figure(figsize=(12, 6))

# Plot training accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()


test_loss, test_acc = model.evaluate(test_dataset)
print(f'test loss: {test_loss}, test_acc: {test_acc}')

# Save the entire model as a SavedModel.
model.save('saved_models/detection_model')
