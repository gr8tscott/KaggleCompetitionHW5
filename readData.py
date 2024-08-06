import pandas as pd
import os

data_dir = './histopathologic-cancer-detection'

ls = os.listdir(data_dir)
print(ls)

sample = pd.read_csv('histopathologic-cancer-detection/sample_submission.csv')
print(sample.shape)
print(sample.head())

train = pd.read_csv('histopathologic-cancer-detection/train_labels.csv', dtype=str)
print(train.shape)
print(train.head())

# train.id = train.id + '.tif'
# train.head()
import matplotlib.pyplot as plt
import cv2

# Load the data
labels = pd.read_csv('histopathologic-cancer-detection/train_labels.csv', dtype=str)
print("labels: ", labels.head())

# Visualize the data
def show_images(images, labels, num=10):
    for i in range(num):
        plt.subplot(2, 5, i+1)
        img_path = os.path.join('histopathologic-cancer-detection/train', images[i] + '.tif')
        img = cv2.imread(img_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Label: {}'.format(labels[i]))
        plt.axis('off')
    plt.show()

show_images(labels['id'].values, labels['label'].values)
print('show images')

#Preprocess the data, data augmentation and normalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print('hit 1.1')
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=90,
    zoom_range=0.2
)
print('hit 1.2')
train_generator = datagen.flow_from_dataframe(
    dataframe=labels,
    directory='train',
    x_col='id',
    y_col='label',
    subset='training',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='binary',
    target_size=(96, 96)
)
print('hit 1.3')
validation_generator = datagen.flow_from_dataframe(
    dataframe=labels,
    directory='train',
    x_col='id',
    y_col='label',
    subset='validation',
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode='binary',
    target_size=(96, 96)
)


#Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
print('hit 2.1')
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])
# print('hit 2.2')
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
print('hit 2.2')
x_batch, y_batch = next(train_generator)
print(x_batch.shape, y_batch.shape)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print('hit 2.3')
#Train the model
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // train_generator.batch_size,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // validation_generator.batch_size,
#     epochs=10
# )
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)


#Evaluate the model performance
# import matplotlib.pyplot as plt
print('hit 3.1')
# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
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

print('hit 3.2')
#Make predictions
test_generator = datagen.flow_from_directory(
    directory='test',
    target_size=(96, 96),
    batch_size=32,
    class_mode=None,
    shuffle=False
)

predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size, verbose=1)

print('hit 3.3 finish')
#Submission
import numpy as np
test_labels = (predictions > 0.5).astype(np.int)
submission = pd.DataFrame({'id': test_generator.filenames, 'label': test_labels})
submission['id'] = submission['id'].str.replace('test/', '').str.replace('.tif', '')
submission.to_csv('submission.csv', index=False)

# Okay that code works and it looks like history compiled. I don't have the submission.csv exporting code or the plot of it though. Is there a way for me to add this code without having to fully rerun 