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
presence_labels = []
dataset_path = os.getcwd() + '/dataset_detection/dataset'

# Load and preprocess images
for filename in os.listdir(dataset_path):
    if filename.endswith(".png"):
        # Extract information from the filename
        match = re.match(r"x(\d+)_y(\d+)_(\d)\.png", filename)
        if match:
            x, y, presence = map(lambda x: int(x) if x.isdigit() else x, match.groups())
            # coordinates.append([x, y])
            coordinates.append([x, y, x + 64, y + 64])

            # Use 'presence' variable to determine the presence of an object
            presence_labels.append(presence)

            # Load and preprocess the image using the presence label
            img = Image.open(os.path.join(dataset_path, filename)).convert('L').resize(
                (BACKGROUND_IMG_SIZE[0], BACKGROUND_IMG_SIZE[1]))
            img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
            img_array = np.expand_dims(img_array, axis=-1)
            images.append(img_array)

# Convert lists to NumPy arrays
X = np.array(images)
y_coordinates = np.array(coordinates)
y_presence = np.array(presence_labels)


# # Adjusted train-test split
# X_train, X_test, y_train_coordinates, y_train_presence, y_test_coordinates, y_test_presence = train_test_split(
#     X, y_coordinates, y_presence, test_size=0.1, random_state=42
# )

# Combine coordinates and presence labels into a single array
y_combined = np.column_stack((y_coordinates, y_presence))

# Split the dataset into training and testing sets
X_train, X_test, y_train_combined, y_test_combined = train_test_split(
    X, y_combined, test_size=0.2, random_state=42
)

# Separate the combined labels back into individual arrays
y_train_coordinates = y_train_combined[:, :-1]  # All columns except the last one
y_train_presence = y_train_combined[:, -1]  # Last column

y_test_coordinates = y_test_combined[:, :-1]
y_test_presence = y_test_combined[:, -1]


# Create TensorFlow datasets with dictionary targets
train_dataset = tf.data.Dataset.from_tensor_slices(
    ({
        'input_layer': X_train  # Assuming 'input_layer' is the name of your input layer
    }, {
        'presence_output': y_train_presence,  # Presence labels for binary classification
        'coordinates_output': y_train_coordinates
    })
)
test_dataset = tf.data.Dataset.from_tensor_slices(
    ({
        'input_layer': X_test
    }, {
        'presence_output': y_test_presence,
        'coordinates_output': y_test_coordinates
    })
)

# Optionally, you can shuffle and batch the datasets
batch_size = 32
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)
test_dataset = test_dataset.batch(batch_size)
"""END OF TEST CHANGES"""
model = models.Sequential()

# Define the input layer
input_layer = layers.Input(shape=BACKGROUND_IMG_SIZE, name='input_layer')

# Convolutional layers
conv1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
maxpool1 = layers.MaxPooling2D((2, 2))(conv1)
conv2 = layers.Conv2D(64, (3, 3), activation='relu')(maxpool1)
maxpool2 = layers.MaxPooling2D((2, 2))(conv2)
conv3 = layers.Conv2D(128, (3, 3), activation='relu')(maxpool2)
maxpool3 = layers.MaxPooling2D((2, 2))(conv3)

# Flatten layer
flatten_layer = layers.Flatten()(maxpool3)

# Dense layers
dense1 = layers.Dense(128, activation='relu')(flatten_layer)

# Output layers
presence_output = layers.Dense(1, activation='sigmoid', name='presence_output')(dense1)
coordinates_output = layers.Dense(4, activation='linear', name='coordinates_output')(dense1)

# Create the model using the Functional API
model = models.Model(inputs=input_layer, outputs=[presence_output, coordinates_output])

# Compile the model with separate loss functions for each output
model.compile(
    optimizer='adam',
    loss={'presence_output': 'binary_crossentropy', 'coordinates_output': 'mean_squared_error'},
    metrics=['accuracy']
)

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
plt.plot(history.history['presence_output_accuracy'], label='Training Accuracy')
plt.plot(history.history['val_presence_output_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['presence_output_loss'], label='Training Loss')
plt.plot(history.history['val_presence_output_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()


# test_loss, test_acc = model.evaluate(test_dataset)
# print(f'test loss: {test_loss}, test_acc: {test_acc}')

# Save the entire model as a SavedModel.
model.save('saved_models/detection_model')
