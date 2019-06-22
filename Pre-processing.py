# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:49:36 2019

@author: fro
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
%matplotlib qt
data = pd.read_csv('kilauea.csv').to_numpy(dtype='float32')
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
    

def dataoversample(cutstart,cutend,dataset):  
    for i in range(len(new_dataset)):
        if new_dataset[i][0] == cutstart:
            start=i
            break
    for i in range(len(new_dataset)):
        if new_dataset[i][0] == cutend:
            end=i
            break
    dataset_to_repair = new_dataset[start:end,:]
    days = 1
    month = 1
    happened = False
    gap=0
    dataset_repaired= []
    Month_toogle = True
    for i in range(len(dataset_to_repair)):
        if i > len(dataset_to_repair)-1: #Avoid exceeding length of array
            break       
        if days == 1:
            if dataset_to_repair[i,1] > 7:
                if Month_toogle:
                    month_days = 30
                    Month_toogle = False
                else:
                    month_days = 31 
                    Month_toogle = True                     
            else:      
                if Month_toogle:
                    month_days = 31
                    Month_toogle = False #Treating each month
                else:
                    month_days = 30
                    Month_toogle = True 
            if dataset_to_repair[i,1] == 2:
                month_days = 28
            if  dataset_to_repair[i,0] % 4 == 0 and dataset_to_repair[i,1] == 2:
                month_days = 29   
        #print('Month =',month,'It Has',month_days)
        if dataset_to_repair[i,2] == days:
            dataset_repaired.append(dataset_to_repair[i,:]) #if Day Exist store the record
            #print('Does Exists day is',days,'/',month_days,'Data to store is ',dataset_to_repair[i,:])
            days+=1
        elif dataset_to_repair[i,2] != days:
            if dataset_to_repair[i,1] != month:
                #print('not equal month')
                gap = abs(days - month_days)+1
                #print('Gap is',gap)
                for c in range(gap):
                    if c == 0:
                        mean_flunct =  (dataset_to_repair[i,3] +  dataset_to_repair[i-1,3])/2
                        mean_heat =  (dataset_to_repair[i,4] +  dataset_to_repair[i-1,4])/2         # if one day is missing just take the mean
                        to_insert = np.array([[dataset_to_repair[i-1,0], month,days,mean_flunct,mean_heat ]], np.float)
                        dataset_repaired.append(to_insert)
                        days+=1
                        #print('Does not exist END',days,'/',month_days,'Data to store is ',to_insert)
                        happened = True
                    else:
                        mean_flunct =  (mean_flunct +  dataset_to_repair[i,3])/2
                        mean_heat =  (mean_heat +  dataset_to_repair[i+1,3])/2    #if there are more days adjust the mean accordingly
                        to_insert = np.array([[dataset_to_repair[i-1,0], month,days,mean_flunct,mean_heat ]], np.float)  
                        dataset_repaired.append(to_insert)
                        days+=1
                        #print('Does not exist Multiple END',days,'/',month_days,'Data to store is ',to_insert)
                days+=1
            if dataset_to_repair[i,1] == month:
                if  happened == True:  
                    gap =  dataset_to_repair[i,2] -2 #if not get the difference between days missing
                    #print('GAP',gap)
                else:
                    gap =  dataset_to_repair[i,2] - days 
                for z in range(int(gap)):
                    if z==0:
                        mean_flunct =  (dataset_to_repair[i,3] +  dataset_to_repair[i-1,3])/2
                        mean_heat =  (dataset_to_repair[i,4] +  dataset_to_repair[i-1,4])/2         # if one day is missing just take the mean
                        to_insert = np.array([[dataset_to_repair[i,0], dataset_to_repair[i,1],days,mean_flunct,mean_heat ]], np.float)
                        dataset_repaired.append(to_insert)
                        #print('Does not exist IN',days,'/',month_days,'Data to store is ',to_insert)
                        days+=1
                    else:
                       # print('Does not exist IN Multiple',days,'/',month_days,'Data to store is ',to_insert)
                        mean_flunct =  (mean_flunct +  dataset_to_repair[i,3])/2
                        mean_heat =  (mean_heat +  dataset_to_repair[i+1,3])/2    #if there are more days adjust the mean accordingly
                        to_insert = np.array([[dataset_to_repair[i,0], dataset_to_repair[i,1],days,mean_flunct,mean_heat ]], np.float)     
                        dataset_repaired.append(to_insert)
                        days+=1   
                gap=0
                if happened == True:
                   # print('inside happened current record',dataset_to_repair[i,:])
                    #print('Records Stored',dataset_to_repair[i-1,:],' and ', dataset_to_repair[i,:],'day is',days+1)
                    dataset_repaired.append(dataset_to_repair[i-1,:])
                    dataset_repaired.append(dataset_to_repair[i,:])
                    happened = False
                    days+=2
                else:
                    dataset_repaired.append(dataset_to_repair[i,:]) #append the current record.   
                    #print('Data Appended outside loops',dataset_to_repair[i,:])
                    days+=1
        if days > month_days: #increasing months and initializing day
            days=1
            month+=1
        if month > 12:
            month = 1
    return dataset_repaired
 
#Sumation of Hotspots
new_dataset = entries_summation(dataset)
new_dataset = np.array(new_dataset,dtype=float)

#Replacing missing values
dataset_repaired = dataoversample(2013,2018,new_dataset)

#Save Dataset into a CSV
for i in range(len(dataset_repaired)):
    dataset_repaired[i] = dataset_repaired[i].flatten() #Flatten arrays saved in the list

#new_dataset = pd.DataFrame(dataset_repaired)
#new_dataset.to_csv("Oversampledkilanuea.csv") #Export to CSV

dataset_repaired = np.array(dataset_repaired,np.object)
for i in range(len(dataset_repaired)):
    dataset_repaired[i,0] = str(int(dataset_repaired[i,0])) +'-'+ str(int(dataset_repaired[i,1]))+'-'+str(int(dataset_repaired[i,2]))

dataset_repaired = np.delete(dataset_repaired, 1, 1) 
dataset_repaired = np.delete(dataset_repaired, 1, 1) 
dataset_repaired = np.delete(dataset_repaired, 2, 1) 
dataset_repaired = pd.DataFrame(dataset_repaired)

dataset_repaired[0] = pd.to_datetime(dataset_repaired[0]) # Covenrt to Date Time
dataset_repaired[1] = dataset_repaired[1].astype(float) # Conver from object to float
dataset_repaired.dtypes

dataset_repaired = dataset_repaired.set_index(0)

#sns.set(rc={'figure.figsize':(11, 4)})
#dataset_repaired[1].plot(linewidth=0.5); #Plotting the Data

to_train= dataset_repaired.loc[:,:]
#to_test= new_dataset.loc['2018-08-07':]

#####Prediction model Starts Here#####
'''
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
'''

'''
#Splitting into training and test set
training_seq = int(len(sequence)*0.80)
test_seq = len(sequence) - training_seq
train,test = sequence[0:training_seq,:],sequence[training_seq:len(sequence),:]

#Transforming the dataset into timesteps selected
n_steps = 14
trainX,trainy =  split_sequence(train,n_steps)
testX,testy = split_sequence(test,n_steps)

#Transforming into the format expected from the LSTM
n_features = 1
trainX = trainX.reshape((trainX.shape[0],trainX.shape[1],n_features))
testX = testX.reshape((testX.shape[0],testX.shape[1],n_features))

#Build and train the neural network
model = Sequential()
model.add(LSTM(20, return_sequences=True,input_shape=(n_steps, 1)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(20,input_shape=(n_steps, 1)))  # returns a sequence of vectors of dimension 32
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')
model.fit(trainX,trainy,epochs=10,shuffle = False)

#Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

#Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainy = scaler.inverse_transform(trainy)

testPredict = scaler.inverse_transform(testPredict)
testy = scaler.inverse_transform(testy)

#Calculate RMSE
trainScore = math.sqrt(mean_squared_error(trainy,trainPredict))
print('Train score : %.2f RMSE' %(trainScore))
testScore = math.sqrt(mean_squared_error(testy,testPredict))
print('Train score : %.2f RMSE' %(testScore))


trainPredictPlot = np.empty_like(dataset_repaired)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[n_steps:len(trainPredict)+n_steps,:] = trainPredict

testPredictPlot = np.empty_like(dataset_repaired)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(trainPredict)+(n_steps*2)+1:len(dataset_repaired)-1,:] = testPredict

'''
sequence = np.array(to_train)

# length of input
input_len = len(sequence)

# The window length of the moving average used to generate
# the output from the input in the input/output pair used
# to train the LSTM
# e.g. if tsteps=2 and input=[1, 2, 3, 4, 5],
#      then output=[1.5, 2.5, 3.5, 4.5]
tsteps = 1

# The input sequence length that the LSTM is trained on for each output point
lahead = 14

# training parameters passed to "model.fit(...)"
batch_size = 1
epochs = 100

# ------------
# MAIN PROGRAM
# ------------

print("*" * 33)
if lahead >= tsteps:
    print("STATELESS LSTM WILL ALSO CONVERGE")
else:
    print("STATELESS LSTM WILL NOT CONVERGE")
print("*" * 33)

# Since the output is a moving average of the input,
# the first few points of output will be NaN
# and will be dropped from the generated data
# before training the LSTM.
# Also, when lahead > 1,
# the preprocessing step later of "rolling window view"
# will also cause some points to be lost.
# For aesthetic reasons,
# in order to maintain generated data length = input_len after pre-processing,
# add a few points to account for the values that will be lost.

to_drop = max(tsteps - 1, lahead - 1)
data_input = dataset_repaired

# set the target to be a N-point average of the input
expected_output = data_input.rolling(window=tsteps, center=False).mean()

# when lahead > 1, need to convert the input to "rolling window view"
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
if lahead > 1:
    data_input = np.repeat(data_input.values, repeats=lahead, axis=1)
    data_input = pd.DataFrame(data_input)
    for i, c in enumerate(data_input.columns):
        data_input[c] = data_input[c].shift(i)

# drop the nan
expected_output = expected_output[to_drop:]
data_input = data_input[to_drop:]

def create_model(stateful):
    model = Sequential()
    model.add(LSTM(20,input_shape=(lahead, 1),batch_size=batch_size,stateful=stateful))
    model.add(Dense(1))
    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mse', optimizer='sgd')
    return model
#Kilauea     sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

print('Creating Stateful Model...')
model_stateful = create_model(stateful=True)


# split train/test data
def split_data(x, y, ratio=0.8):
    to_train = int(input_len * ratio)
    # tweak to match with batch_size
    to_train -= to_train % batch_size

    x_train = x[:to_train]
    y_train = y[:to_train]
    x_test = x[to_train:]
    y_test = y[to_train:]

    # tweak to match with batch_size
    to_drop = x.shape[0] % batch_size
    if to_drop > 0:
        x_test = x_test[:-1 * to_drop]
        y_test = y_test[:-1 * to_drop]

    # some reshaping


    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)
'''
If you rescale the whole dataset before train_test_split, there is a chance of introducing look-ahead bias. I think you should split the data first, then use the scaler you trained on the training dataset to transform the test dataset.
'''

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

scaler = MinMaxScaler(feature_range=(0,1))
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
y_train = scaler.fit_transform(y_train)
y_test = scaler.fit_transform(y_test)

x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_test = x_test.reshape((x_test.shape[0],x_test.shape[1],1))
y_train = y_train.reshape((y_train.shape[0],1))
y_test = y_test.reshape((y_test.shape[0],1))

#Stateful: In the stateful LSTM configuration,
 #internal state is only reset when the reset_state()
 #function is called.
 
print('Training')
val_loss_history = []
loss_history=[]
for i in range(epochs):
    print('Epoch', i + 1, '/', epochs)
    # Note that the last state for sample i in a batch will
    # be used as initial state for sample i in the next batch.
    # Thus we are simultaneously training on batch_size series with
    # lower resolution than the original series contained in data_input.
    # Each of these series are offset by one step and can be
    # extracted with data_input[i::batch_size].
    #model_stateful.compile(loss='binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
    history_stateful = model_stateful.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=1,
                       verbose=1,
                       validation_data=(x_test, y_test),
                       shuffle=False)
    val_loss_history.append(history_stateful.history['val_loss'])
    loss_history.append(history_stateful.history['loss'])
    model_stateful.reset_states()
    
print('Predicting')
predicted_stateful = model_stateful.predict(x_test, batch_size=batch_size)

plt.plot(loss_history)
plt.plot(val_loss_history)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper right')
plt.show()

test_predict = scaler.inverse_transform(predicted_stateful)
test_y_groundtruth = scaler.inverse_transform(y_test)

#Stateless: In the stateless LSTM configuration,
 #internal state is reset after each training batch 
 #or each batch when making predictions.
 
print('Creating Stateless Model...')
model_stateless = create_model(stateful=False)

print('Training')
history_stateless = model_stateless.fit(x_train, y_train,batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    shuffle=False)

print('Predicting')
predicted_stateless = model_stateless.predict(x_test, batch_size=batch_size)

plt.plot(history_stateless.history['loss'])
plt.plot(history_stateless.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','validation'],loc='upper right')
plt.show()

print('Plotting Results')
#plt.subplot(3, 1, 1)
plt.plot(y_test)
plt.title('Results')
#plt.subplot(3, 1, 2)
# drop the first "tsteps-1" because it is not possible to predict them
# since the "previous" timesteps to use do not exist
plt.plot((predicted_stateful).flatten()[tsteps - 1:])
#plt.title('Stateful: Predicted')
#plt.subplot(3, 1, 3)
plt.plot((predicted_stateless).flatten())
#plt.title('Stateless: Predicted')
plt.legend(['Expected','Stateful','Stateless'],loc='upper right')
plt.show()

