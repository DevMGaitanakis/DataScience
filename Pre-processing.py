# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:49:36 2019

@author: fro
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib qt
data = pd.read_csv('Etnapart2.csv').to_numpy(dtype='float32')
dataset =np.flipud(data)

def data_missing(dataset):
    counter = 1
    months_count=0
    for i in range(len(new_dataset)):
        while counter != new_dataset[i,1]:
            if counter == 12:
                counter =1
            else:
                months_count +=1
                counter+=1
                print(new_dataset[i,:])
        if i+1 > len(new_dataset)-1:
            break
        elif counter+1 == new_dataset[i+1,1]:
            counter+=1
    return months_count
    
def entries_summation(dataset):
    r=0
    entries_to_sum,to_sum1,to_sum2= [],[],[]
    new_dataset=[]
    for i in range(len(dataset)):
        if i+1 > len(dataset)-1: #check if loop exceeds the length of the dataset if yes append the last items for summation
            to_sum1.append(dataset[i][3])
            entries_to_sum.append(to_sum1)
            new_dataset.append(dataset[i])
            new_dataset[len(new_dataset)-1][3] = sum(entries_to_sum[len(entries_to_sum)-1])/len(entries_to_sum[len(entries_to_sum)-1])
            break
        if dataset[i][2] == dataset[i+1][2]:
            to_sum1.append(dataset[i][3])
            r+=1
        elif dataset[i][2] != dataset[i+1][2] and r > 1 : # Handles multiple day entries
            to_sum1.append(dataset[i][3]) #Append previous item to the list
            entries_to_sum.append(to_sum1) # Insert everything
            new_dataset.append(dataset[i]) #Create new List with the Dataset Entry
            new_dataset[len(new_dataset)-1][3] = sum(entries_to_sum[len(entries_to_sum)-1])/len(entries_to_sum[len(entries_to_sum)-1]) #Update Excess radiation with the summation of the pixels identified this day
            to_sum1 = []        
            r=0
        elif dataset[i][2] != dataset[i+1][2] and r == 1 : # Handles two day entries
            to_sum2.append(to_sum1[0])
            to_sum2.append(dataset[i][3])
            new_dataset.append(dataset[i])
            entries_to_sum.append(to_sum2)
            new_dataset[len(new_dataset)-1][3] = sum(entries_to_sum[len(entries_to_sum)-1])/len(entries_to_sum[len(entries_to_sum)-1]) #Update Sum        
            to_sum1,to_sum2 = [],[]
            r=0
        elif dataset[i][2] != dataset[i+1][2] and r == 0 : #Handles one day entry
            entries_to_sum.append(dataset[i][3])
            new_dataset.append(dataset[i])
    return (new_dataset)
    
new_dataset = entries_summation(dataset)

new_dataset = np.array(new_dataset,dtype=float)

for i in range(len(new_dataset)):
    new_dataset[i,0] = str(int(new_dataset[i,0])) +'-'+ str(int(new_dataset[i,1]))+'-'+str(int(new_dataset[i,2]))

new_dataset = np.delete(new_dataset, 1, 1) 
new_dataset = np.delete(new_dataset, 1, 1) 
new_dataset = np.delete(new_dataset, 2, 1) 
new_dataset = pd.DataFrame(new_dataset)



new_dataset[0] = pd.to_datetime(new_dataset[0]) # Covenrt to Date Time
new_dataset[1] = new_dataset[1].astype(float) # Conver from object to float
new_dataset.dtypes
new_dataset = new_dataset.set_index(0)

to_train= new_dataset.loc[:'2017-07-12']
to_test= new_dataset.loc['2018-08-07':]

sns.set(rc={'figure.figsize':(11, 4)})
new_dataset[1].plot(linewidth=0.5);

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

def split_sequence(sequence,n_steps):
    X,y = list(),list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix],sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

sequence = np.array(to_train)
from sklearn import preprocessing
sequence1 =preprocessing.scale(sequence)
n_steps = 7
X,y =  split_sequence(sequence,n_steps)

n_features = 1
X = X.reshape((X.shape[0],X.shape[1],n_features))

model = Sequential()
model.add(LSTM(1, return_sequences=True,
               input_shape=(n_steps, 1)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(100, return_sequences=True)) 
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100, return_sequences=True)) # returns a sequence of vectors of dimension 32
model.add(LSTM(100))  # return a single vector of dimension 32
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
model.fit(X,y,epochs=50)

x_predict = np.array(to_test)
x_predict= x_predict[0:7]

x_predict = x_predict.reshape(1,n_steps,n_features)
yhat = model.predict(x_predict)
print(yhat)



