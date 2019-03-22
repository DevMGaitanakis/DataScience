# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:32:28 2019

@author: fro
"""

import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.metrics import classification_report

def showcontents(imgs_dataset):
    for i in range(64):
        ax = plt.subplot(8, 8, i+1)
        ax.axis('off')
        plt.imshow(imgs_dataset[i], cmap='Greys')       
        
def replace(labels):
    a=[]
    for i in range(48,58):
        a.append(i) 
    for i in range(65,91):
        a.append(i) 
    a.append(97)
    a.append(98)
    for i in range(100,105):
        a.append(i) 
    a.append(110)
    a.append(113)
    a.append(114)
    a.append(116)
    replace = np.array(a)    
    for i in range(np.size(np.unique(labels))):
        labels = np.where(labels == i ,replace[i], labels)
    return labels

def ANN_struct(x_test_data,x_train_data):
    x_test_data = x_test_data.reshape(x_test_data.shape[0], 784) # reshapping data CNN expects 4 by 4 array
    x_train_data = x_train_data.reshape(x_train_data.shape[0], 784)
    input_shape = (784)
    return x_test_data,x_train_data,input_shape

def ANN(x_train_data,y_train_labels,x_test_data,y_test_labels,input_shape):
    x_test_data = x_test_data.reshape(x_test_data.shape[0], input_shape) # reshapping data CNN expects 4 by 4 array
    x_train_data = x_train_data.reshape(x_train_data.shape[0], input_shape)
    Image_Size = 784
    model = Sequential()
    model.add(Dense(units=32, activation='sigmoid', input_shape=(Image_Size,)))
    model.add(Dense(units=Total_Classes, activation='softmax'))
    model.summary()
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train_data, y_train_labels, batch_size=128, epochs=10, verbose=False, validation_split=.1)
    loss, accuracy  = model.evaluate(x_test_data, y_test_labels, verbose=False)
    x_test_data1= np.expand_dims(x_test_data[0], axis=0)
    pred = model.predict(x_test_data1)
    pred.argmax()
    return history,loss,accuracy,model

def CNN_struct(x_test_data,x_train_data,model):
    x_test_data = x_test_data.reshape(x_test_data.shape[0], 28, 28, 1) # reshapping data CNN expects 4 by 4 array
    x_train_data = x_train_data.reshape(x_train_data.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    return x_test_data,x_train_data,input_shape

def CNN(x_train_data,y_train_labels,x_test_data,y_test_labels,input_shape):
    model = Sequential()
    model.add(Conv2D(28, (3, 3), input_shape=input_shape, activation="relu"))#Convolution Step Applying feature detectors to generate a Feature Map
    model.add(MaxPooling2D(pool_size=(2, 2)))#Take the Feature map and take maximum values in order to create a Pooled Feature map
    model.add(Conv2D(28, (3, 3), activation="relu"))#(3,3) dimensions of feature detectors
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Flatten Pooling into 1 column array
    model.add(Dense(128, activation ='relu')) # Hidden Layers
    model.add(Dense(Total_Classes, activation='softmax')) # Output Layer
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    loss, accuracy  = model.evaluate(x_test_data, y_test_labels, verbose=False)
    history = model.fit(x=x_train_data,y=y_train_labels, epochs=10)
    return history,loss,accuracy,model

def eval_CNN(x_test_data,model):
    data_to_predict= x_test_data.reshape(x_test_data.shape[0],28,28) 
    pred = []
    for i in range(np.size(data_to_predict,axis = 0)):
        predictions = model.predict(data_to_predict[i].reshape(1,28,28,1))
        pred.append(predictions.argmax())
    lab = np.unique(y_test_labels_replaced)
    target_names = ["Class "+ chr(lab[i]) for i in range(Total_Classes)]
    print(classification_report(y_test_labels_initial, pred, target_names=target_names))
    
def eval_ANN(x_test_data,model):
    data_to_predict= x_test_data.reshape(x_test_data.shape[0],784) 
    pred = []
    for i in range(np.size(data_to_predict,axis=0)):
        predictions = model.predict(np.expand_dims(x_test_data[i], axis=0))
        pred.append(predictions.argmax())
    lab = np.unique(y_test_labels_replaced)
    target_names = ["Class "+ chr(lab[i]) for i in range(Total_Classes)]
    print(classification_report(y_test_labels_initial, pred, target_names=target_names))

    
x_test_init = np.array(pd.read_csv('emnist-balanced-test.csv'),dtype= np.uint8) #importing datasets
x_train_init =np.array(pd.read_csv('emnist-balanced-train.csv'),dtype= np.uint8)

y_test_labels = x_test_init[:,0] #extracting y labels out
y_test_labels_initial = x_test_init[:,0] #taking the remaining dataset
y_test_labels_replaced = replace(y_test_labels) #storing the true values that correspond to letters to a new array
x_test_data = x_test_init[:,1:]

y_train_labels = x_train_init[:,0]
x_train_data = x_train_init[:,1:]

x_test_data,x_train_data,input_shape = ANN_struct(x_test_data,x_train_data)

#x_test_data,x_train_data,input_shape = CNN_struct(x_test_data,x_train_data)

x_train_data = x_train_data.astype('float32') #normalizing data for faster processing
x_test_data = x_test_data.astype('float32') 
x_train_data /= 255
x_test_data /= 255

Total_Classes = np.unique(y_train_labels).size #total classes of the dataset

y_train_labels = keras.utils.to_categorical(y_train_labels, Total_Classes) # Y label categories are beign encoded into binary
y_test_labels = keras.utils.to_categorical(y_test_labels, Total_Classes)# so the model will not be able to make unessesary relations

history,loss,accuracy,model = CNN(x_train_data,y_train_labels,x_test_data,y_test_labels)
eval_CNN(x_test_data,model)

#history,loss,accuracy,model = ANN(x_train_data,y_train_labels,x_test_data,y_test_labels,input_shape)
#eval_ANN(x_test_data,model)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()
print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')

#images = x_test_data.reshape(x_test_data.shape[0],28,28) 
#showcontents(images)
