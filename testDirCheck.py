import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

def load_test_images(image_dir, img_size):
    images = []
    filenames = []
    for file_name in os.listdir(image_dir):
        if file_name.endswith('.tif'):
            img_path = os.path.join(image_dir, file_name)
            image = load_img(img_path, target_size=img_size)
            image = img_to_array(image) / 255.0  # Normalize
            images.append(image)
            filenames.append(file_name)
        else:
            print(f"Skipping file: {file_name} (not a .tif file)")
    
    if not images:
        print(f"No images found in directory: {image_dir}")
    
    return np.array(images), filenames

img_size = (64, 64)  # Use a smaller image size for testing
print('hit 12')
# Path to test data
test_dir = 'histopathologic-cancer-detection/test'
print("Loading Test Images...")
test_images, filenames = load_test_images(test_dir, img_size)

# Verify test images were loaded
print(f"Loaded {len(test_images)} test images.")

if len(test_images) > 0:
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
else:
    print("No test images found. Please check the test directory.")
