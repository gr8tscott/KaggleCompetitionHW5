import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

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
print('hit 1.4')
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
print('hit 1.5')
# Check the first batch
x_batch, y_batch = next(train_generator)
print(x_batch.shape, y_batch.shape)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
print('hit 1.6')
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print('hit 1.7')
# Fit the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)
print('hit 1.8')