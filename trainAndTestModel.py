import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

print('hit 1')

# Function to prepare data generators
def prepare_generators(train_dir, labels_file, img_size, batch_size):
    labels_df = pd.read_csv(labels_file)
    labels_df['id'] = labels_df['id'].astype(str) + '.tif'
    labels_df['label'] = labels_df['label'].astype(str)
    
    print('Dataframe head:\n', labels_df.head())

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=train_dir,
        x_col='id',
        y_col='label',
        subset='training',
        class_mode='binary',
        target_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    
    validation_generator = datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=train_dir,
        x_col='id',
        y_col='label',
        subset='validation',
        class_mode='binary',
        target_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    
    print('Train generator length:', len(train_generator))
    print('Validation generator length:', len(validation_generator))

    return train_generator, validation_generator

# Path to training data and labels
train_dir = 'histopathologic-cancer-detection/train'
labels_file_path = 'histopathologic-cancer-detection/train_labels.csv'
img_size = (96, 96)  # Use a smaller image size for testing
batch_size = 32

# Prepare data generators
print('Preparing generators...')
train_generator, validation_generator = prepare_generators(train_dir, labels_file_path, img_size, batch_size)

# Build a simple CNN model
print('Building the model...')
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
print("Training the Model...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Evaluate the model performance
print("Evaluating the model performance")
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Load and preprocess test data
def load_test_images(image_dir, img_size):
    datagen = ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_directory(
        directory=image_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    return test_generator

# Path to test data
test_dir = 'histopathologic-cancer-detection/test'
print("Loading Test Images...")
test_generator = load_test_images(test_dir, img_size)

# Predict on the test data
print("Making Predictions...")
predictions = model.predict(test_generator)
predictions_binary = (predictions > 0.5).astype(int)

# Prepare the submission file
print("Preparing Submission File...")
sample_submission_path = 'histopathologic-cancer-detection/sample_submission.csv'
sample_submission = pd.read_csv(sample_submission_path)
sample_submission['label'] = predictions_binary
sample_submission.to_csv('submission.csv', index=False)

print("Submission file created: 'submission.csv'")
