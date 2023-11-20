"""
Traffic Signs Classification based on Convolutional Neural Network.
"""

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from support import (
    create_dataset,
    extract_image_and_label,
    create_test_and_validation_dataset,
    show_images,
)

# Read file names from directory
list_dataset = tf.data.Dataset.list_files(os.getcwd() + r'/dataset_augmented/*/*', shuffle=True)
ds_validation = tf.data.Dataset.list_files(os.getcwd() + r'/dataset_validation/*/*', shuffle=True)

# Create labeled dataset from list of files
labeled_dataset = list_dataset.map(extract_image_and_label)


print(f"Dataset length: {len(labeled_dataset)}\n")

train_size = int(0.7 * len(labeled_dataset))

ds_train = labeled_dataset.take(train_size)
ds_test = labeled_dataset.skip(train_size)

batch_size = 64
ds_train = ds_train.batch(batch_size)
ds_test = ds_test.batch(batch_size)

for image, label in ds_train.take(1):
    print("Image shape:", image.shape)
    print("Label:", label)

for image, label in ds_test.take(1):
    print("Image shape:", image.shape)
    print("Label:", label)

# Define model for signs classification
signs_classification_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((3, 3)),
    tf.keras.layers.Conv2D(128, (2, 2), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])


# Compile model with ADAM optimizer
signs_classification_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train model
history = signs_classification_model.fit(
    ds_train,
    epochs=2,
    validation_data=ds_test,
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = signs_classification_model.evaluate(ds_test)
print(f'test loss: {test_loss}, test_acc: {test_acc}')

# Extract images and labels from the validation dataset
validation_images, validation_labels = zip(*[(image, label) for image, label in ds_validation.map(extract_image_and_label)])

# Convert the extracted images and labels to numpy arrays
validation_images = np.array(validation_images)
validation_labels = np.array(validation_labels)

# Make predictions using the model
predictions = signs_classification_model.predict(validation_images, batch_size=batch_size)

print(f'Prediction shape: {predictions.shape}')

# Evaluate the model performance
evaluation_result = signs_classification_model.evaluate(validation_images, validation_labels, batch_size=batch_size)
print(f'Evaluation result: {evaluation_result}')

# Convert predicted probabilities to predicted labels
predicted_labels = np.argmax(predictions, axis=1)

# Create the confusion matrix using TensorFlow
conf_matrix = tf.math.confusion_matrix(validation_labels, predicted_labels)

# Extract values from TensorFlow tensor
conf_matrix = conf_matrix.numpy()

# Display the confusion matrix using a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
