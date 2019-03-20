# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:42:22 2019

@author: fro
"""

import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
# Preparing the dataset
# Setup train and test splits
#(x_train, y_train), (x_test, y_test) = emnist.load_data()
x_test = np.array(pd.read_csv('emnist-balanced-test.csv'),dtype= np.uint8)
y_test1 = x_test[:,0]
x_test1 = x_test[:,1:]

x_train =np.array(pd.read_csv('emnist-balanced-train.csv'),dtype= np.uint8)
# Making a copy before flattening for the next code-segment which displays images
data= open('emnist-balanced-test-images-idx3-ubyte')


from scipy import io as sio
mat = sio.loadmat('emnist-letters.mat')
data = mat['dataset']


X_train = mat['dataset'][0][0][0][0][0][0]
y_train = mat['dataset'][0][0][0][0][0][1]
X_test = data['test'][0,0]['images'][0,0]
y_test = data['test'][0,0]['labels'][0,0]

image_size = 784 # 28 x 28
X_test1 = x_test1.reshape(x_test1.shape[0],28,28) 
x_test = x_test.reshape(x_test.shape[0], image_size)

# Convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


for i in range(64):
    ax = plt.subplot(8, 8, i+1)
    ax.axis('off')
    plt.imshow(X_test1[i], cmap='Greys')

model = Sequential()
# The input layer requires the special input_shape parameter which should match
# the shape of our training data.
model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.summary()
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=False, validation_split=.1)
loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)


classifier = Sequential()
classifier.add(Dense(6, input_shape=(image_size,), activation ='relu',kernel_initializer='uniform'))
classifier.add(Dense(6, activation ='relu',kernel_initializer='uniform'))
classifier.add(Dense(1, activation ='softmax')) ##softmax if more than one dependant variables
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=False, validation_split=.1)
loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')