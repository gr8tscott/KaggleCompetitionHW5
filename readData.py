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