#Author: Malav Shah
# Train image classification using Tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import split_folders
from tensorflow import optimizers
from tensorflow.keras.optimizers import SGD
import matplotlib as plt
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt

def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


#directory with all image
data_dir = os.path.join(os.getcwd(), 'images')
subdirs = [x[1] for x in os.walk(data_dir)]

#all classes to classify an image
classes = subdirs[0]

#split data into train/val/test
split_folders.ratio('images', output = 'split_set', seed = 1337, ratio =(0.6, 0.4, 0))

#split into different train/val/test directories
data_dir = os.path.join(os.getcwd(), 'split_set')
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

total_train = 0
total_val = 0
#find total training sizes
for c in classes:
    temp_url = os.path.join(train_dir, c)
    total_train += len(os.listdir(temp_url))
    temp_url = os.path.join(val_dir, c)
    total_val += len(os.listdir(temp_url))

print(total_train)
print(total_val)

#training variables
batch_size = 40
epochs = 50
IMG_HEIGHT = 150
IMG_WIDTH = 150

#image processing generator (add different horizontal vertical scaling if overfitting/underfitting)
train_image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=45, zoom_range = 0.5, width_shift_range=.15, height_shift_range=.15)
validation_image_generator = ImageDataGenerator(rescale=1./255)

#get images from train and validatino directories and process them
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size, directory=train_dir, shuffle=True,target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='sparse', color_mode = 'rgb')
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size, directory=val_dir, shuffle=True,target_size=(IMG_HEIGHT, IMG_WIDTH), class_mode='sparse', color_mode = 'rgb')

#visualize some images
sample_training_images, _ = next(train_data_gen)
#plotImages(sample_training_images[:5])

#ML model with multiple layers (change depending on performance)
model = Sequential()
model.add(Flatten()) 
model.add(Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.3))) 
#model.add(Dropout(0.5)) 
model.add(Dense(50, activation=tf.keras.layers.LeakyReLU(alpha=0.3))) 
#model.add(Dropout(0.3))
model.add(Dense(25, activation=tf.keras.layers.LeakyReLU(alpha=0.3))) 
#model.add(Dropout(0.1))
model.add(Dense(5, activation= 'softmax'))

model.compile(  loss='sparse_categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

#save to tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#train the model
history = model.fit(
    train_data_gen,
    steps_per_epoch= total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size,
    callbacks=[tensorboard_callback]
)


#visualize model data -> loss and accuracy to see over and under fitting
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()