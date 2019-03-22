# -*- coding: utf-8 -*-
"""CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FXsIV707u_hbp4SK_jcBKXGxDmPBoTiK
"""

#Libraries
from keras.models import Sequential #Initialize CNN
from keras.layers import Conv2D #First Step ADD convolution Layer
from keras.layers import MaxPooling2D # Step Two Pooling Layers
from keras.layers import Flatten # Step three Converting Pooling Layers into a large feature vector
from keras.layers import Dense # Adds Layers
from keras.preprocessing.image import ImageDataGenerator
from google.colab import drive

drive.mount("/content/gdrive/")

#Building CNN
classifier = Sequential() #Initialize CNN
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))#Convolution Step Applying feature detectors to generate a Feature Map
classifier.add(MaxPooling2D(pool_size=(2, 2)))#Take the Feature map and take maximum values in order to create a Pooled Feature map

classifier.add(Conv2D(32, (3, 3), activation="relu"))#(3,3) dimensions of feature detectors
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten()) # Flatten Pooling into 1 column array
classifier.add(Dense(128, activation ='relu')) # Hidden Layers
classifier.add(Dense(1, activation='sigmoid')) # Output Layer
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fit CNN to the images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
       'gdrive/My Drive/dataset/training_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'gdrive/My Drive/dataset/test_set',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000
        ) #images of test set

#make new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('gdrive/My Drive/dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)

print(result)
print(training_set.class_indices)