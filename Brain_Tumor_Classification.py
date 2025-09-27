# Brain Tumor Classification using Convolutional Neural Network (CNN)
# This model classifies MRI images into different tumor classes.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Rescaling
from keras.models import Sequential

# Step 1: Set the folder path for your dataset
folder_path = r"C:\Users\akjee\Documents\AI\DL\CNN\Brain Tumor"

# Step 2: Load training and validation datasets from directories
# The images are resized to 64x64 and loaded in batches of 32
train_data = tf.keras.utils.image_dataset_from_directory(
    folder_path + r'\Training',
    validation_split=0.2,         # 20% of data for validation
    subset="training",            # This is the training subset
    seed=123,                     # Seed for reproducibility
    image_size=(64, 64),          # Resize images to 64x64
    batch_size=32                 # Batch size
)

val_data = tf.keras.utils.image_dataset_from_directory(
    folder_path + r'\Training',
    validation_split=0.2,         # 20% of data for validation
    subset="validation",          # This is the validation subset
    seed=123,                     # Seed for reproducibility
    image_size=(64, 64),          # Resize images to 64x64
    batch_size=32                 # Batch size
)

# Step 3: Get class names (labels)
class_names = train_data.class_names
print("Classes found:", class_names)

# Step 4: Optimize data pipeline for performance
autotune = tf.data.AUTOTUNE
train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=autotune)
val_data = val_data.cache().prefetch(buffer_size=autotune)

# Step 5: Build the CNN model
model = Sequential([
    Rescaling(1./255, input_shape=(64, 64, 3)),         # Normalize pixel values
    Conv2D(32, (3, 3), activation='relu'),              # First convolutional layer
    MaxPool2D((2, 2)),                                 # First max pooling layer
    Conv2D(64, (3, 3), activation='relu'),              # Second convolutional layer
    MaxPool2D((2, 2)),                                 # Second max pooling layer
    Flatten(),                                         # Flatten feature maps to 1D
    Dense(128, activation='relu'),                     # Dense layer
    Dropout(0.5),                                      # Dropout for regularization
    Dense(len(class_names), activation='softmax')       # Output layer for multi-class classification
])

# Step 6: Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',             # Use sparse categorical crossentropy for integer labels
    metrics=['accuracy']
)

# Step 7: Show model summary
model.summary()

# Step 8: Train the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=25                                           # You can increase epochs for better results
)

# Step 9: Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 10: Evaluate the model on validation data
print("Model Evaluation on Validation Data:")
model.evaluate(val_data)