import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.io import imread

# Load the labels
train_labels = pd.read_csv('histopathologic-cancer-detection/train_labels.csv')

# Plot class distribution
train_labels['label'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Display some sample images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes = axes.ravel()

for i in range(2):
    sample = train_labels[train_labels['label'] == i].sample(1)
    img_path = os.path.join('histopathologic-cancer-detection/train', sample['id'].values[0] + '.tif')
    img = imread(img_path)
    axes[i].imshow(img)
    axes[i].set_title(f'Label: {i}')
plt.show()
