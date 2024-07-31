import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
print('hit 1')
# Define the path to the data
data_path = 'histopathologic-cancer-detection'
train_dir = os.path.join(data_path, 'train')
test_dir = os.path.join(data_path, 'test')
labels_file = os.path.join(data_path, 'train_labels.csv')
print('hit 1.1')
# Load your dataframe
dataframe = pd.read_csv(labels_file)

# Ensure the dataframe is correct
print(dataframe.head())

# Verify image paths
print(os.listdir(train_dir)[:5])  # Print the first 5 files in the train directory
print('hit 1.2')
# Add the .tif extension to the ids in the dataframe
dataframe['id'] = dataframe['id'] + '.tif'

# Convert labels to strings
dataframe['label'] = dataframe['label'].astype(str)

# Define the ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
print('hit 1.3')
# Create the train generator
train_generator = datagen.flow_from_dataframe(
    dataframe=dataframe,
    directory=train_dir,
    x_col='id',
    y_col='label',
    subset='training',
    class_mode='binary',
    target_size=(96, 96),
    batch_size=32,
    shuffle=True
)
print('hit 4')
# Create the validation generator
validation_generator = datagen.flow_from_dataframe(
    dataframe=dataframe,
    directory=train_dir,
    x_col='id',
    y_col='label',
    subset='validation',
    class_mode='binary',
    target_size=(96, 96),
    batch_size=32,
    shuffle=True
)
print('hit 5')
# Load pre-trained model
base_model = VGG16(input_shape=(96, 96, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=x)
print('hit 6')
# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
print('hit 7')
# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
print('hit 8')
# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=1, #set it from 10 to 1 to get to completion
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping]
)
print('hit 9')
# Evaluate the model performance
print('Evaluating the model performance')
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

print('hit 10')
# Make predictions
test_generator = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(96, 96),
    batch_size=32,
    class_mode=None,
    shuffle=False
)
print('hit 11')
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size, verbose=1)
print('hit 12')
# Submission
test_labels = (predictions > 0.5).astype(np.int)
submission = pd.DataFrame({'id': test_generator.filenames, 'label': test_labels[:, 0]})
submission['id'] = submission['id'].str.replace('test/', '').str.replace('.tif', '')
submission.to_csv('submission.csv', index=False)
print('hit 13 finish')
print('Submission file created: submission.csv')
