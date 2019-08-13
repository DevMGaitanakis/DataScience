from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers

%matplotlib inline
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
            new_dataset[len(new_dataset)-1][3] = sum(entries_to_sum[len(entries_to_sum)-1])
            break
        if dataset[i][2] == dataset[i+1][2]:
            to_sum1.append(dataset[i][3])
            r+=1
        elif dataset[i][2] != dataset[i+1][2] and r > 1 : # Handles multiple day entries
            to_sum1.append(dataset[i][3]) #Append previous item to the list
            entries_to_sum.append(to_sum1) # Insert everything
            new_dataset.append(dataset[i]) #Create new List with the Dataset Entry
            new_dataset[len(new_dataset)-1][3] = sum(entries_to_sum[len(entries_to_sum)-1]) #Update Excess radiation with the summation of the pixels identified this day
            to_sum1 = []        
            r=0
        elif dataset[i][2] != dataset[i+1][2] and r == 1 : # Handles two day entries
            to_sum2.append(to_sum1[0])
            to_sum2.append(dataset[i][3])
            new_dataset.append(dataset[i])
            entries_to_sum.append(to_sum2)
            new_dataset[len(new_dataset)-1][3] = sum(entries_to_sum[len(entries_to_sum)-1]) #Update Sum        
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

def data_distribution(dataset):
    array1 = np.array([])  
    array2 = np.array([]) 
    rangeof_values = {}    
    range_start =0
    range_end = 100
    for z in range(60):
        key =  str(range_start)+'to'+ str(range_end)
        rangeof_values[key] = 0 
        for i in range (len(dataset)):
            if dataset[i] > range_start and dataset[i] < range_end:
               rangeof_values[key] +=1
        range_start += 100
        range_end += 100
    for k, v in rangeof_values.items():
        array1 = np.append(array1,k)
        array2 = np.append(array2,v)
    return array1,array2

def data_prep_plot(dataset,path,name):
    dataset = np.array(dataset,np.object)
    for i in range(len(dataset)):
        dataset[i,0] = str(int(dataset[i,0])) +'-'+ str(int(dataset[i,1]))+'-'+str(int(dataset[i,2]))
    dataset = np.delete(dataset, 1, 1) 
    dataset = np.delete(dataset, 1, 1) 
    dataset = np.delete(dataset, 2, 1) 
    dataset = pd.DataFrame(dataset)   
    dataset[0] = pd.to_datetime(dataset[0]) # Covenrt to Date Time
    dataset[1] = dataset[1].astype(float) # Covenrt from object to float 
    dataset = dataset.set_index(0)
    sns.set(rc={'figure.figsize':(11, 4)})
    dataset[1].plot(linewidth=0.5); #Plotting the Data
    plt.savefig(path+'\\'+fname+'.png')
    return dataset

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

def CustomLSTM(layers,lahead,lr,epochs,batch_size,units,dropout,shuffle):
    program.append('model_LSTM = Sequential()')
    if layers == 1:
        program.append('model_LSTM.add(LSTM(units =' + str(units) + ',return_sequences = False, input_shape=(' + str(lahead) + ',1)))')
    else:
        program.append('model_LSTM.add(LSTM(units =' + str(units) + ',return_sequences = True, input_shape=(' + str(lahead) + ',1)))')  
        program.append('model_LSTM.add(Dropout(' + str(dropout) + '))')
        for i in range(layers-1):
            if i+1 ==(layers-1):
                program.append('model_LSTM.add(LSTM(units ='+ str(units) +'))')
                program.append('model_LSTM.add(Dropout(' + str(dropout) + '))')
            else:
                unit_inside =  'units' + str(i+1)
                unit_inside = unit_inside
                program.append('model_LSTM.add(LSTM(units ='+ str(units) + ',return_sequences = True))')
                program.append('model_LSTM.add(Dropout(' + str(dropout) + '))')
    program.append('model_LSTM.add(Dense(units=1))')
    program.append('RMSprop = optimizers.RMSprop(lr=' + str(lr) + ', rho=0.9, epsilon=None, decay=0.0)')
    program.append('model_LSTM.compile(optimizer =RMSprop ,loss="mean_squared_error",metrics=["mse"])')
    program.append('history = model_LSTM.fit(train_x,train_y,epochs=' + str(epochs) + ',batch_size=' + str(batch_size) + ',shuffle=' + str(shuffle) + ',validation_split =0.2)')     
    return program

def CustomGRU(layers,lahead,lr,epochs,batch_size,units,dropout,shuffle):
    program.append('model_GRU = Sequential()')
    if layers == 1:
        program.append('model_GRU.add(GRU(units =' + str(units) + ',return_sequences = False, input_shape=(' + str(lahead) + ',1)))')
    else:
        program.append('model_GRU.add(GRU(units =' + str(units) + ',return_sequences = True, input_shape=(' + str(lahead) + ',1)))')  
        program.append('model_GRU.add(Dropout(' + str(dropout) + '))')
        for i in range(layers-1):
            if i+1 ==(layers-1):
                program.append('model_GRU.add(LSTM(units ='+ str(units) +'))')
                program.append('model_GRU.add(Dropout(' + str(dropout) + '))')
            else:
                unit_inside =  'units' + str(i+1)
                unit_inside = unit_inside
                program.append('model_GRU.add(GRU(units ='+ str(units) + ',return_sequences = True))')
                program.append('model_GRU.add(Dropout(' + str(dropout) + '))')
    program.append('model_GRU.add(Dense(units=1))')
    program.append('RMSprop = optimizers.RMSprop(lr=' + str(lr) + ', rho=0.9, epsilon=None, decay=0.0)')
    program.append('model_GRU.compile(optimizer =RMSprop ,loss="mean_squared_error",metrics=["mse"])')
    program.append('history = model_GRU.fit(train_x,train_y,epochs=' + str(epochs) + ',batch_size=' + str(batch_size) + ',shuffle=' + str(shuffle) + ',validation_split =0.2)')     
    return program


#Time period that is going to be extracted and oversampled / this section is only for plotting purposes

'''
cutstart = 2000
cutend = 2019
for i in range(len(new_dataset)):
    if new_dataset[i][0] == cutstart:
        start=i
        break
for i in range(len(new_dataset)):
    if new_dataset[i][0] == cutend:
        end=i
        break
to_plot = new_dataset[start:end,:]
#to_plot = data_prep_plot(to_plot,path,fname)
#Data Distribution
labels_distribution,values_distribution = data_distribution(np.array(dataset_not_op))


  #Getting rid of columns that are not required for the prediction and plotting
        dataset_repaired = data_prep_plot(dataset_repaired,path,fname)
    #Data Distribution
        # labels_distribution,values_distribution = data_distribution(np.array(dataset_repaired))
 '''       
       
#files = ['kilauea','Erebus','ErtaAle','Nyiragongo']

#periods_to_extract = [[2013,2018],[2003,2010],[2006,2017],[2005,2018]]

data = pd.read_csv('Nyiragongo.csv')
dataset =np.flipud(data)
new_dataset = entries_summation(dataset)
new_dataset = np.array(new_dataset,dtype=float)


dataset_not_op = new_dataset[:,3]
sequence = np.array(dataset_not_op)

dataset_repaired = dataoversample(2005,2018,new_dataset)
for i in range(len(dataset_repaired)):
    dataset_repaired[i] = dataset_repaired[i].flatten() #Flatten arrays saved in the list    
dataset_repaired = np.array(dataset_repaired)    
dataset_repaired = dataset_repaired[:,3]
sequence = np.array(dataset_repaired)


sequence = sequence.reshape(-1, 1)
input_len = len(sequence)
#Splitting into training and test set
training_seq = int(len(sequence)*0.90)
test_seq = len(sequence) - training_seq
train,test = sequence[0:training_seq,:],sequence[training_seq:len(sequence),:]

scaler = MinMaxScaler(feature_range=(0,1))
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)
train_x,train_y =  split_sequence(train,30)
train_y = train_y.reshape(-1,1,1)
test_x,test_y = split_sequence(test,30)
test_y = test_y.reshape(-1,1,1)

model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(30,1), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(RepeatVector(1))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(optimizer='RMSprop', loss='mse')
plot_model(model, show_shapes=False, to_file='predict_lstm_autoencoder.png')
history = model.fit(train_x,train_y,epochs=50,batch_size=10,shuffle=False,validation_split =0.2)

plt.figure(figsize=(10,7))       
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig(path+'\\'+fname+'Results.png')
        
prediction = model.predict(test_x)
actual = test_y.reshape(-1,1)
prediction = prediction.reshape(-1,1)
plt.figure(figsize=(10,7)) 
plt.plot(actual,linewidth=1.0)
plt.plot(prediction,linewidth=2.0)
plt.title('Predictions')
plt.ylabel('Excess Radiation')
plt.xlabel('Days')
plt.legend(['Actual', 'Prediction'], loc='upper right')
plt.savefig(path+'\\'+fname+'predictions.png')
print (np.max(history.history['loss']))
print (np.min(history.history['loss']))
print (np.max(history.history['val_loss']))
print (np.min(history.history['val_loss']))
print("MSE:"+ str(mean_squared_error(actual,prediction)))

f = open(path+'\\results',"a+")
f.write( fname + ' Results \n')
f.write('Max Training Loss: ' + str(np.max(history.history['loss']))+'\n')
f.write('Max Training Loss: ' + str(np.min(history.history['loss']))+'\n')
f.write('Max Training Loss: ' + str(np.max(history.history['val_loss']))+'\n')
f.write('Max Training Loss: ' + str(np.min(history.history['val_loss']))+'\n')
f.write('MSE: ' +  str(mean_squared_error(actual,prediction) )+'\n')
f.close()
plt.close("all")


for file in files:
    data = pd.read_csv(file+'.csv')
    dataset =np.flipud(data)
    new_dataset = entries_summation(dataset)
    new_dataset = np.array(new_dataset,dtype=float)
    dataset_not_op = new_dataset[:,3]
    dataset_repaired = dataoversample(periods_to_extract[files.index(file)][0],periods_to_extract[files.index(file)][1],new_dataset)
    for i in range(len(dataset_repaired)):
        dataset_repaired[i] = dataset_repaired[i].flatten() #Flatten arrays saved in the list    
    dataset_repaired = np.array(dataset_repaired)    
    dataset_repaired = dataset_repaired[:,3]
    #Getting rid of columns that are not required for the prediction and plotting
        #dataset_repaired = data_prep_plot(dataset_repaired,path,fname)
    #Data Distribution
        # labels_distribution,values_distribution = data_distribution(np.array(dataset_repaired))
    #Save Dataset into a CSV    
    new_dataset = pd.DataFrame(new_dataset)
    new_dataset.to_csv("Oversampledkilanuea.csv") #Export to CSV
    hyper_parameters = [
    #[1,30,0.001,50,10,30,0,False],
    #[1,30,0.001,50,10,30,0,True],
    #[1,30,0.01,50,10,30,0,False],        
    #[1,60,0.01,50,10,60,0,False],          
    #[1,30,0.001,50,10,30,0,False],      
    #[1,60,0.001,50,10,60,0,False],  
    #[2,30,0.01,50,10,30,0.2,False],     
    #[2,60,0.01,50,10,60,0.2,False],  
    #[2,30,0.001,50,10,30,0.2,False],     
    #[2,60,0.001,50,10,60,0.2,False],
    #[3,30,0.01,50,10,30,0.2,False],     
    #[3,60,0.01,50,10,60,0.2,False],  
    #[3,30,0.001,50,10,30,0.2,False],     
    #[3,60,0.001,50,10,60,0.2,False],  
    [4,30,0.01,50,10,30,0.2,False],      
    [4,60,0.01,50,10,60,0.2,False],  
    [4,30,0.001,50,10,30,0.2,False],     
    [4,60,0.001,50,10,60,0.2,False]                       
    ]
    for l in range(len(hyper_parameters)):
        print('Experiment' + str(l))
        if l == 0:
            sequence = np.array(dataset_not_op)
        else:
            sequence = np.array(dataset_repaired)
        #sequence = np.array(new_dataset[:,3])
        sequence = sequence.reshape(-1, 1)
        input_len = len(sequence)
        #Splitting into training and test set
        training_seq = int(len(sequence)*0.90)
        test_seq = len(sequence) - training_seq
        train,test = sequence[0:training_seq,:],sequence[training_seq:len(sequence),:]
        
        scaler = MinMaxScaler(feature_range=(0,1))
        train = scaler.fit_transform(train)
        test = scaler.fit_transform(test)
        train_x,train_y =  split_sequence(train,int(hyper_parameters[l][1]))
        test_x,test_y = split_sequence(test,hyper_parameters[l][1])
        
        #reshaping to the the format required by the LSTM
        train_x = train_x.reshape((train_x.shape[0],train_x.shape[1],1))
        test_x = test_x.reshape((test_x.shape[0],test_x.shape[1],1))

        program = []
        program = CustomLSTM(hyper_parameters[l][0],hyper_parameters[l][1],hyper_parameters[l][2],hyper_parameters[l][3],hyper_parameters[l][4],hyper_parameters[l][5],hyper_parameters[l][6],hyper_parameters[l][7])
        #program = CustomGRU(hyper_parameters[l][0],hyper_parameters[l][1],hyper_parameters[l][2],hyper_parameters[l][3],hyper_parameters[l][4],hyper_parameters[l][5],hyper_parameters[l][6],hyper_parameters[l][7])
       
        for z in range(len(program)):
            exec(program[z])
            
        path = 'C:\\Users\\fro\\Desktop\\All_volcanoes\\Results_' + str.lower(file)
        fname = 'Experiment' + str(l)  
    
        plt.figure(figsize=(15,10))       
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model train vs validation loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig(path+'\\'+fname+'Results.png')
    
        prediction = model_LSTM.predict(test_x)
        actual = test_y
        
        plt.figure(figsize=(15,8))
        plt.plot(actual,linewidth=1.0)
        plt.plot(prediction,linewidth=2.0)
        plt.title('Predictions')
        plt.ylabel('Excess Radiation')
        plt.xlabel('Days')
        plt.legend(['Actual', 'Prediction'], loc='upper right')
        plt.savefig(path+'\\'+fname+'predictions.png')
        
        f = open(path+'\\results',"a+")
        f.write( fname + ' Results \n')
        f.write('Max Training Loss: ' + str(np.max(history.history['loss']))+'\n')
        f.write('Max Training Loss: ' + str(np.min(history.history['loss']))+'\n')
        f.write('Max Training Loss: ' + str(np.max(history.history['val_loss']))+'\n')
        f.write('Max Training Loss: ' + str(np.min(history.history['val_loss']))+'\n')
        f.write('MSE: ' +  str(mean_squared_error(actual,prediction) )+'\n')
        f.close()
        plt.close("all")
        
