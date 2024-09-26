from sklearn.model_selection import train_test_split
import os
import random

imagedir = '/Users/lly/codes/datasets/Pascal_voc_test_data/data_dataset_voc/JPEGImages'
outdir = '/Users/lly/codes/datasets/Pascal_voc_test_data/data_dataset_voc/'
os.makedirs(outdir, exist_ok=True)

images = []
for file in os.listdir(imagedir):
    filename = file.split('.')[0]
    images.append(filename)
random.shuffle(images)

train_size = 0.8
val_size = 0.2

train, test = train_test_split(images, test_size=val_size, random_state=0)

with open(os.path.join(outdir, 'train.txt'), 'w') as f:
    f.write('\n'.join(train))

with open(os.path.join(outdir, 'val.txt'), 'w') as f:
    f.write('\n'.join(test))