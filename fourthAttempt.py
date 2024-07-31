# Import necessary libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

print('hit 1')
# Function to load images and labels in batches
def load_images_in_batches(image_dir, labels_file, img_size, batch_size):
    labels_df = pd.read_csv(labels_file)
    image_ids = labels_df['id'].tolist()
    labels = labels_df['label'].tolist()
    
    num_images = len(image_ids)
    images = []
    labels_list = []

    for i in range(0, num_images, batch_size):
        batch_ids = image_ids[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        for image_id, label in zip(batch_ids, batch_labels):
            img_path = os.path.join(image_dir, f"{image_id}.tif")
            if os.path.isfile(img_path):
                image = load_img(img_path, target_size=img_size)
                image = img_to_array(image) / 255.0  # Normalize
                images.append(image)
                labels_list.append(label)
                
        # Convert to numpy array after each batch
        yield np.array(images), np.array(labels_list)
        images = []
        labels_list = []
print('hit 2')

# Load a small subset for testing
def load_small_subset(image_dir, labels_file, img_size, subset_size):
    labels_df = pd.read_csv(labels_file).head(subset_size)
    image_ids = labels_df['id'].tolist()
    labels = labels_df['label'].tolist()
    
    images = []
    labels_list = []

    for image_id, label in zip(image_ids, labels):
        img_path = os.path.join(image_dir, f"{image_id}.tif")
        if os.path.isfile(img_path):
            image = load_img(img_path, target_size=img_size)
            image = img_to_array(image) / 255.0  # Normalize
            images.append(image)
            labels_list.append(label)

    return np.array(images), np.array(labels_list)

print('hit 3')
# Path to training data and labels
train_dir = 'histopathologic-cancer-detection/train'
labels_file_path = 'histopathologic-cancer-detection/train_labels.csv'
img_size = (64, 64)  # Use a smaller image size for testing

print('hit 4')
# Load training images and labels
print("Loading Training Images and Labels...")
batch_size = 1000
train_images = []
train_labels = []

print('hit 5')
for X_batch, y_batch in load_images_in_batches(train_dir, labels_file_path, img_size, batch_size):
    train_images.extend(X_batch)
    train_labels.extend(y_batch)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
print('hit 6')

# Create a validation set from the training data
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

print('hit 7')
# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

print('hit 8')
# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print('hit 9')
# Print model summary
model.summary()

print('hit 10')
# Train the model
print("Training the Model...")
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val)
)

print('hit 11')
# Load and preprocess test data
def load_test_images(image_dir, img_size):
    images = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.tif'):
            img_path = os.path.join(image_dir, file_name)
            image = load_img(img_path, target_size=img_size)
            image = img_to_array(image) / 255.0  # Normalize
            images.append(image)
    return np.array(images)

print('hit 12')
# Path to test data
test_dir = 'histopathologic-cancer-detection/test'
print("Loading Test Images...")
test_images = load_test_images(test_dir, img_size)


# Predict on the test data
print("Making Predictions...")
predictions = model.predict(test_images)

# Convert predictions to binary
predictions_binary = (predictions > 0.5).astype(int)

# Prepare the submission file
print("Preparing Submission File...")
sample_submission_path = 'histopathologic-cancer-detection/sample_submission.csv'
sample_submission = pd.read_csv(sample_submission_path)
sample_submission['label'] = predictions_binary
sample_submission.to_csv('submission.csv', index=False)

print("Done! Submission file created: 'submission.csv'")