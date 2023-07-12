import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

data_path = r'C:\Users\Dell 7567\Desktop\vehicle_dataset'

# Load the images and labels
images = []
labels = []

for label in range(10):
    label_path = os.path.join(data_path, str(label))
    for image_file in os.listdir(label_path):
        if image_file.endswith('.png'):
            image_path = os.path.join(label_path, image_file)
            image = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(32, 32))
            image = tf.keras.preprocessing.image.img_to_array(image)
            images.append(image)
            labels.append(label)

images = np.array(images)
labels = np.array(labels)

# Normalize pixel values
images = images / 255.0

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Undersample the majority class (class 0 - Sedan)
class_counts = np.bincount(y_train)
min_class_count = np.min(class_counts[1:])  # Ignore the majority class
undersampled_indices = []

for label in range(10):
    indices = np.where(y_train == label)[0]
    undersampled_indices.extend(np.random.choice(indices, size=min_class_count, replace=False))

x_train_undersampled = x_train[undersampled_indices]
y_train_undersampled = y_train[undersampled_indices]

# Build the model
model = keras.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the training parameters
batch_size = 32
epochs = 12

# Train the model and store the training history
history = model.fit(x_train_undersampled, y_train_undersampled, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

# Plot the training process

# Plot the training process
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Process')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model on the validation set
y_pred = model.predict(x_val)
y_pred = np.argmax(y_pred, axis=1)

confusion_mtx = confusion_matrix(y_val, y_pred)
print(confusion_mtx)

# Calculate accuracy, precision, recall, and F1 score
accuracy = np.mean(y_pred == y_val)

# Generate the classification report
class_names = ["Sedan", "SUV", "Pickup truck", "Van", "Box truck",
               "Motorcycle", "Flatbed truck", "Bus", "Pickup truck w/ trailer", "Flatbed truck w/ trailer"]
report = classification_report(y_val, y_pred, target_names=class_names)

# Print the report
print("Performance Report:\n")
print(f"Accuracy: {accuracy}\n")
print(report)
