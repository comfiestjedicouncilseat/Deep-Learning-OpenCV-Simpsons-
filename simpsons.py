import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# Defining constants
IMG_SIZE = (80, 80)
channels = 1
char_path = r"C:\Users\Michael\Documents\Coding" \
            r"\DeepLearningOpenCV(Simpsons)\simpsons_dataset"

# Creating a dictionary for all the characters in the dataset
char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))

# Sort the dictionary by number of images in each character folder
char_dict = caer.sort_dict(char_dict, descending=True)

# Selects the 10 characters with the most number of images
characters = []
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >= 10:
        break

# Creating the training data
# Accesses a specific file path, adds all images for each element in the
# 'characters' list to the training set
# For preprocessing the data
train = caer.preprocess_from_dir(char_path, characters, channels=channels,
                                 IMG_SIZE=IMG_SIZE, isShuffle=True)

# Display an image from the training set, with a resolution of 30 by 30
# Also displays the image in gray scale
plt.figure(figsize=(30, 30))
plt.imshow(train[0][0], cmap="gray")
plt.show()

# Extracting the feature set and labels from each element in the training set
# Essentially "seperates" the training set into a feature set and labels
# Feature set is a 4 dimenional tensor, can be fed into model
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

# Normalizing the feature set to be in the range of (0, 1)
featureSet = caer.normalize(featureSet)

# Convert labels from integers to binary class vectors
labels = to_categorical(labels, len(characters))

# Create training and validation data
# Train on the training data, test on the validation data
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels,
                                                      val_ratio=0.2)

# Clear up some memory by deleting uneccessary variables
del train
del featureSet
del labels
gc.collect()

# Image generator
# Will create new image based on existing data
# Instantiate image generator
BATCH_SIZE = 32
EPOCHS = 10
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

# Creating the model, will compile the model as well
model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels,
                                          output_dim=len(characters),
                                          loss="binary_crossentropy",
                                          decay=1e-6, learning_rate=0.001,
                                          momentum=0.9, nesterov=True)

print(model.summary())
