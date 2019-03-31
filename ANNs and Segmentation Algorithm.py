import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Activation 
from keras.layers import Input,merge,Reshape,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
import seaborn as sns 
import time
import cv2
# display sample images from inported emnist dataset
def showcontents(imgs_dataset):
    for i in range(64):
        ax = plt.subplot(8, 8, i+1)
        ax.axis('off')
        plt.imshow(imgs_dataset[i], cmap='Greys')
# replace the the label indexes from 1-47 to show binary codes to convert to actual labels (e.g. a,b....x,y,z)
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

# Multilayer Percepton Neural Network 
def MLP(x_train_data,y_train_labels,input_shape, epochs):
    model = Sequential()
    # hidden layer with 512 neurons, initial layer must be preset as takes inputs as image
    model.add(Dense(units=512, kernel_initializer = 'uniform', input_shape=(input_shape,)))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    #model.add(Dropout(rate = 0.5)) can use drop out to improve prediction no change on EMNIST dataset
    # hidden layer with 512 neurons each
    ################ comment out if not needed#############
    model.add(Dense(units=512))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    #model.add(Dropout(rate = 0.5))
    # hidden layer with 512 neurons each
    ################ comment out if not needed#############
    model.add(Dense(units=512))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    # hidden layer with 512 neurons each
    ################ comment out if not needed#############
    model.add(Dense(units=512))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    #######################################################
    # output layer with 47 neurons
    model.add(Dense(units=Total_Classes, activation='softmax'))
    model.summary()
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train_data, y_train_labels, batch_size=128, epochs=epochs, verbose=False, validation_split = .2)### validation_set = .2 20% of trainign data used for valdiation
    return history, model

# this model is modified VGGNet 16layer to be used with 28x28 input images
def CNN(x_train_data,y_train_labels,x_test_data,y_test_labels,input_shape, epochs):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding="same"))# activation="relu", Convolution Step Applying feature detectors to generate a Feature Map
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))#pooling to 14x14x64
    
    model.add(Conv2D(128, (3, 3),padding="same"))#(3,3) kernel dimensions. layer mapping to 14x14x128
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(128, (3, 3),padding="same"))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))# pooling to 7x7x128
    
    model.add(Conv2D(256, (3, 3),padding="same"))# map to 256 feature maps 7x7x256
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(256, (3, 3),padding="same"))#(3,3) dimensions of feature detectors
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(Conv2D(256, (3, 3),padding="same"))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))# pooling to 3x3x256
    
    model.add(Flatten()) # Flatten Pooling into 1 column array
    model.add(Dense(200, activation ='relu')) # Hidden Layers 200 neurons
    model.add(Dense(200, activation ='relu')) # Hidden Layers 200 neurons
    model.add(Dense(Total_Classes, activation='softmax')) # Output Layer no neurons = total_classes
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    history = model.fit(x=x_train_data,y=y_train_labels, epochs=epochs,  validation_data=(x_test_data, y_test_labels))
    loss, accuracy  = model.evaluate(x_test_data, y_test_labels, verbose=False)
    return history,loss,accuracy,model

def CNN3(x_train_data,y_train_labels,input_shape, epochs):
    model = Sequential()
    # 2 Convolutional Layers 28x28x64
    model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding="same")) # Convolution Step Applying feature detectors to generate a Feature Map 28x28x64
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # Pooling to 14x14x128
    model.add(Conv2D(128, (3, 3),padding="same"))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # Pooling to 7x7x256
    model.add(Conv2D(256, (3, 3),padding="same"))
    model.add(BatchNormalization())
    model.add(Activation(activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) # Pooling to 3x3x256
    #fourth layer
    #model.add(Conv2D(512, (3, 3),padding="same"))#(3,3) dimensions of feature detectors
    #model.add(BatchNormalization())
    #model.add(Activation(activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2))) # Pooling to 3x3x256
    model.add(Flatten()) # Flatten Pooling into array
    # 2 fully Connected layers with 200 neurons each
    model.add(Dense(200, activation ='relu')) # Hidden Layers
    #model.add(Dense(200, activation ='relu')) # Hidden Layers
    #output layer consciting of 47 neurons using softmax
    model.add(Dense(Total_Classes, activation='softmax')) # Output Layer
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    history = model.fit(x=x_train_data,y=y_train_labels, epochs=epochs)# validation_split=0.2 to test on 20% of trainign set)
    #loss, accuracy  = model.evaluate(x_test_data, y_test_labels, verbose=False)
    return history,model
#function used to evaluate Convolutional neural networks
def eval_CNN(x_test_data,model,y_test_labels_initial,y_test_labels_replaced):
    data_to_predict= x_test_data.reshape(x_test_data.shape[0],28,28) 
    pred = []
    for i in range(np.size(data_to_predict,axis = 0)):
        predictions = model.predict(data_to_predict[i].reshape(1,28,28,1))
        pred.append(predictions.argmax())
    lab = np.unique(y_test_labels_replaced)
    accuracy = metrics.accuracy_score(y_test_labels_initial,pred)
    target_names = ["Class "+ chr(lab[i]) for i in range(Total_Classes)]
    print(classification_report(y_test_labels_initial, pred, target_names=target_names))
    return accuracy, pred
#function used to evaluate multilayer perceptron 
def eval_MLP(x_test_data,model, y_test_labels_initial,y_test_labels_replaced):
    data_to_predict= x_test_data.reshape(x_test_data.shape[0],784) 
    pred = []
    for i in range(np.size(data_to_predict,axis=0)):
        predictions = model.predict(np.expand_dims(x_test_data[i], axis=0))
        pred.append(predictions.argmax())
    lab = np.unique(y_test_labels_replaced)
    accuracy = metrics.accuracy_score(y_test_labels_initial,pred)
    target_names = ["Class "+ chr(lab[i]) for i in range(Total_Classes)]
    print(classification_report(y_test_labels_initial, pred, target_names=target_names))
    return accuracy, pred
### evaluate encoder classification performance
def eval_AUTO(x_test_data,model, y_test_labels_initial,y_test_labels_replaced):
    data_to_predict= x_test_data.reshape(x_test_data.shape[0],28,28) 
    pred = []
    for i in range(np.size(data_to_predict,axis = 0)):
        predictions = model.predict(data_to_predict[i].reshape(1,28,28,1))
        pred.append(predictions.argmax())
    lab = np.unique(y_test_labels_replaced)
    accuracy = metrics.accuracy_score(y_test_labels_initial,pred)
    target_names = ["Class "+ chr(lab[i]) for i in range(Total_Classes)]
    print(classification_report(y_test_labels_initial, pred, target_names=target_names))
    return accuracy, pred

#### structure data for CNN and Autoencoder
def CNN_struct(x_test_data,x_train_data):
    x_test_data = x_test_data.reshape(x_test_data.shape[0], 28, 28, 1) # reshapping data CNN expects 4 by 4 array
    x_train_data = x_train_data.reshape(x_train_data.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)
    return x_test_data,x_train_data,input_shape
# function used to process data from initial imported formato to required format separating labels form features etc.    
def norm_data_separate_labels(x_test_init, x_train_init):
    y_test_labels = x_test_init[:,0] #extracting y labels out
    y_test_labels_initial = x_test_init[:,0] #taking the remaining dataset
    y_test_labels_replaced = replace(y_test_labels) #storing the true values that correspond to letters to a new array
    x_test_data = x_test_init[:,1:]
    y_train_labels = x_train_init[:,0]
    x_train_data = x_train_init[:,1:]
    x_train_data = x_train_data.astype('float32') #normalizing data for faster processing
    x_test_data = x_test_data.astype('float32') 
    x_train_data /= 255
    x_test_data /= 255
    Total_Classes = np.unique(y_train_labels).size #total classes of the dataset
    return x_train_data, y_train_labels, x_test_data, y_test_labels, Total_Classes, y_test_labels_initial, y_test_labels_replaced
#######################################################################################################################
######### Autoencoder based on three layer CNN
#######################################################################################################################
    
def encoder(input_img):
    conv1 = Conv2D(64, (3, 3), padding='same')(input_img) #28 x 28 x 64
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation = 'relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 64
    conv2 = Conv2D(128, (3, 3), padding='same')(pool1) #14 x 14 x 128
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation = 'relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 128
    conv3 = Conv2D(256, (3, 3), padding='same')(pool2) #7 x 7 x 256
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation = 'relu')(conv3)
    return conv3

def decoder(conv3):    
    up1 = UpSampling2D((2,2))(conv3) #14 x 14 x 256
    conv6 = Conv2D(128, (3, 3), padding='same')(up1)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation(activation = 'relu')(conv6)
    up2 = UpSampling2D((2,2))(conv6) #14 x 14 x 128
    conv7 = Conv2D(64, (3, 3), padding='same')(up2)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation(activation = 'relu')(conv7)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv7) # 28 x 28 x 1
    return decoded

def fully_connected(enco):
    flat = Flatten()(enco)
    den = Dense(200, activation='relu')(flat)
    out = Dense(Total_Classes, activation='softmax')(den)
    return out
####################################################################################################################
################### autoencoder based on modified VGGNet
#################################################################################################
def encoder1(input_img):
    conv1 = Conv2D(64, (3, 3), padding='same')(input_img) #28 x 28 x 64
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation(activation = 'relu')(conv1)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv1) #28 x 28 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation(activation = 'relu')(conv2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2) #14 x 14 x 64
    conv3 = Conv2D(128, (3, 3), padding='same')(pool1) #14 x 14 x 128
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation(activation = 'relu')(conv3)
    conv4 = Conv2D(128, (3, 3), padding='same')(conv3) #14 x 14 x 128
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation(activation = 'relu')(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4) #7 x 7 x 128
    conv5 = Conv2D(256, (3, 3), padding='same')(pool2) #7 x 7 x 256
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation(activation = 'relu')(conv5)
    conv6 = Conv2D(256, (3, 3), padding='same')(conv5) #7 x 7 x 256
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation(activation = 'relu')(conv6)
    conv8 = Conv2D(512, (3, 3), padding='same')(conv6) #7 x 7 x 512
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation(activation = 'relu')(conv8)
    return conv8

def decoder1(conv8):    
    conv9 = Conv2D(256, (3, 3), padding='same')(conv8) #7 x 7 x 256
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation(activation = 'relu')(conv9)
    conv10 = Conv2D(256, (3, 3), padding='same')(conv9) #7 x 7 x 256
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation(activation = 'relu')(conv10)
    up1 = UpSampling2D((2,2))(conv10) #14 x 14 x 256
    conv11 = Conv2D(128, (3, 3), padding='same')(up1)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation(activation = 'relu')(conv11)
    conv12 = Conv2D(128, (3, 3), padding='same')(conv11)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation(activation = 'relu')(conv12)
    up2 = UpSampling2D((2,2))(conv12) #14 x 14 x 128
    conv13 = Conv2D(64, (3, 3), padding='same')(up2)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation(activation = 'relu')(conv13)
    conv14 = Conv2D(64, (3, 3), padding='same')(conv13)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation(activation = 'relu')(conv14)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(conv14) # 28 x 28 x 1
    return decoded


def fully_connected1(enco):
    flat = Flatten()(enco)
    den = Dense(200, activation='relu')(flat)
    den1 = Dense(200, activation='relu')(den)
    out = Dense(Total_Classes, activation='softmax')(den1)
    return out
##############################################################################
# Confusion matrix plot
def conf_matrix(y_test_labels_initial,predicted_classes):
    cnf_matrix=confusion_matrix(y_test_labels_initial,predicted_classes)
    sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5,fmt='g')
    plt.title("Confusion_Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    return cnf_matrix
##############################################################################
##############################################################################

#importing train and test datasets emnist dataset import
x_test_init = np.array(pd.read_csv('emnist-balanced-test.csv', header = None),dtype= np.uint8) 
x_train_init =np.array(pd.read_csv('emnist-balanced-train.csv', header = None),dtype= np.uint8)

#own dataset import
#x_train_init =np.array(pd.read_csv('training_data.csv', header = None),dtype= np.uint8)
#x_test_init =np.array(pd.read_csv('training_data.csv', header = None),dtype= np.uint8)

# normalizing data, separating data from the labels and getting a number of total classes
x_train_data, y_train_labels, x_test_data, y_test_labels, Total_Classes, y_test_labels_initial, y_test_labels_replaced = norm_data_separate_labels(x_test_init, x_train_init)

# encode labels into binary
y_train_labels = keras.utils.to_categorical(y_train_labels, Total_Classes) # Y label categories are beign encoded into binary
y_test_labels = keras.utils.to_categorical(y_test_labels, Total_Classes)# so the model will not be able to make unessesary relations

###### use this dataset prep for CNN and Convolutional AutoEncoder
x_test_data_c,x_train_data_c,input_shape_c = CNN_struct(x_test_data,x_train_data)

'''
to display the graphs for training epoch losses and accuracies use this
'''
plt.plot(mlp_history.history['acc'], color = 'blue') #1layer  'blue', 2 layer 'red' 3 layer 'magenta'
plt.plot(mlp_history.history['val_acc'], color = 'green')# i layer green, 2 layer 'cyan', 3 layer 'yellow'
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()

''' Run CNN training in this section'''
cnn_epochs = 2
#### train CNN
cnn3_start = time.time()
cnn3_history,cnn3_model = CNN3(x_train_data_c,y_train_labels,input_shape_c, cnn_epochs)
cnn3_end = time.time()
cnn3_train_time = cnn3_end - cnn3_start
#### Evaluate CNN
cnn3_start_eval = time.time()
cnn3_accuracy, cnn3_predicted_classes = eval_CNN(x_test_data_c,cnn3_model,y_test_labels_initial,y_test_labels_replaced)
cnn3_end_eval = time.time()
cnn3_eval_time = cnn3_end_eval - cnn3_start_eval
### CNN confusio matrix
cnn_cm = conf_matrix(y_test_labels_initial, cnn3_predicted_classes)
'''

to run modified vggnet amend the run training section to use the right function
'''

''' Run MLP in this section'''
input_shape = (784)
mlp_epochs = 2

mlp_start = time.time()
mlp_history, mlp_model = MLP(x_train_data,y_train_labels,input_shape, mlp_epochs)
mlp_end = time.time()
mlp_train_time = mlp_end - mlp_start

##### evaluate MLP
mlp_start_eval = time.time()
mlp_accuracy, mlp_pred = eval_MLP(x_test_data,mlp_model, y_test_labels_initial, y_test_labels_replaced)
mlp_end_eval = time.time()
mlp_eval_time = mlp_end_eval - mlp_start_eval

#display confusion matrix
mlp_cm = conf_matrix(y_test_labels_initial, mlp_pred)

###### get training accuracies and losses for each epoch validation must be used to extract validation values
mlp_accuracy = mlp_history.history['acc']
mlp_val_accuracy = mlp_history.history['val_acc']
mlp_loss = mlp_history.history['loss']
mlp_val_loss = mlp_history.history['val_loss']
'''
Autoencoder part here
'''
epochs = 2
input_img = Input(shape = (input_shape_c))
batch_size = 64

#Set auto encoder for training
autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', optimizer = 'adam')
#view architecture
autoencoder.summary()

#tarin autoencoder (encoder decoder part)
start = time.time()
autoencoder_train = autoencoder.fit(x_train_data_c, x_train_data_c, epochs=epochs, verbose=1)
end = time.time()
train_time_ae_train = end - start

########## setup autoecoder clasifier with fully connected layers
encode = encoder(input_img)
full_model = Model(input_img,fully_connected(encode))

### set fully conected veights for the required layers same as encoder decoder part
for l1,l2 in zip(full_model.layers[:12],autoencoder.layers[0:12]):
    l1.set_weights(l2.get_weights())
# set encoder part veights not trainable
for layer in full_model.layers[0:12]:
    layer.trainable = False

#Compile the fully connected model
full_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
full_model.summary()
#Train the fuly connected model to learn the lables and classification
classify_train = full_model.fit(x_train_data_c, y_train_labels, epochs=epochs,verbose=1)#, batch_size=64

autoencoder_accuracy, autoencoder_predicted_classes = eval_AUTO(x_test_data_c,full_model,y_test_labels_initial,y_test_labels_replaced)


'''
Import image of own text and segment it
'''
image = cv2.imread("pred2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.GaussianBlur(gray, (7, 7), 2) add blur if needed
ret,thresh= cv2.threshold(gray ,100,255,cv2.THRESH_BINARY_INV)
#ret,thresh= cv2.threshold(gray ,180,255,cv2.THRESH_TOZERO_INV) allow for different thresholds

# dilate the white portions
dilate = cv2.dilate(thresh, None, iterations=1)
plt.imshow(dilate)
def segmentation(arr):
    list = []
    for i in range(len(arr)):
        if sum(arr[i])== 0:
            list.append(i)
        else:
            list.append(0)   
    nlist=[]

    try:
        if list[0] < list[1]:
            for i in range(1,len(list)):
                if list[i] > list[i+1] or list[i] == 0 and list[i+1] > list[i] :
                    nlist.append(i)
    except IndexError:
        pass  
    return list,nlist   
   
list,nlist = segmentation(dilate)
list2 = []
suma = 0
row = 1
rows = {}

for i in range (0,len(nlist),2):
    for x in range(len(dilate[1])):
        for z in range(nlist[i],nlist[i+1]):
            suma += dilate[z][x]
        if suma == 0:
            list2.append(x)
        else:
            list2.append(0)
            rows[row] = list2       
        suma = 0
    row+=1 
    list2 =[]   
  
nlist2=[]
letterpos = {}

for y in range(1,len(rows)+1):
    if rows[y][0] < rows[y][1]:
        for i in range(1,len(rows[1])):
            try:
                if rows[y][i] > rows[y][i+1]:
                    nlist2.append(rows[y][i])
                    letterpos[y] = nlist2
                elif rows[y][i] < rows[y][i+1] and rows[y][i] == 0 :
                    nlist2.append(rows[y][i+1])
                    letterpos[y] = nlist2
            except IndexError:
                pass   
    nlist2 = [] 

all_letters = np.zeros(shape=(1,28,28))

rows=len(letterpos)
posa=0  
notempty=[]
notemptySize=0
inc=0

#### print rectangles on original images to show segmentation
for a in range(1,rows+1):
    for b in range (0,len(letterpos[a]),2):
        try:
           print('One',letterpos[a][b], 'two' ,letterpos[a][b+1])
           print('Nlist',nlist[inc],nlist[inc+1])
           rowtop,rowbottom = nlist[inc],nlist[inc+1]
           letterfront,letterend= letterpos[a][b],letterpos[a][b+1]
           new = np.array(dilate[rowtop:rowbottom,letterfront:letterend])
           noUse,trimmed = segmentation(new)
           notempty.append(trimmed)
           if len(notempty[notemptySize]) == 0:
               cv2.rectangle(image,pt1=(letterpos[a][b],nlist[inc]),pt2=(letterpos[a][b+1],nlist[inc+1]),color=(0,255,0),thickness=5)
           elif len(notempty[notemptySize]) > 1:
               new = new[trimmed[0]:trimmed[1],:]
               cv2.rectangle(image,pt1=(letterpos[a][b],nlist[inc]+notempty[posa][0]),pt2=(letterpos[a][b+1],nlist[inc+1]-abs(notempty[posa][1] - len(noUse))),color=(0,255,0),thickness=5)
                         
           elif len(notempty[notemptySize]) ==1 :
               new = new[trimmed[0]:,:]
               cv2.rectangle(image,pt1=(letterpos[a][b],nlist[inc]+notempty[posa][0]),pt2=(letterpos[a][b+1],nlist[inc+1]),color=(0,255,0),thickness=5)
           new = np.pad(new, (30, 30), 'minimum')
           resized_image = cv2.resize(new, (28, 28))
           resized_image2 = np.expand_dims(resized_image, axis=0)
           all_letters = np.append(all_letters,resized_image2,axis=0)
           notemptySize+=1
        except IndexError:
                        pass           
        posa+=1      
    inc+=2
all_letters = np.delete(all_letters,[0],axis=0)

### show original image with rectangles around segmented text
plt.imshow(image)

##### use this to export segmented text as csv file if required
ready_exp = all_letters.reshape(all_letters.shape[0],784)
np.savetxt('imagedata.csv', ready_exp, delimiter=',')
### displaye segented letters one by one
showcontents(all_letters)
#normalize data form 0 to 1
all_letters /= 255
# use labels defined for impoted note
andrius_labels = np.array([10,43,38,45,18,30,28,18,28,41,24,18,43,41,46,24,41,39,46,22,24,45,39,46,42,36,43,9,0,18,43,46,42,18,28,12,24,30,45,28,39,32,24,45,20])
eman_labels = np.array([24,38,45,32,41,23,20,30,43,29,20,45,43,48,26,38,15,44,20,41,33,41,11,0,20,45,48,44,20,30,15,26,32,47,30,41,34,26,47,22])

# predict imported note using cnn and CNN trained on own handwritting
pred = []
for i in range(np.size(all_letters,axis = 0)):
        predictions = cnn3_model.predict(all_letters[i].reshape(1,28,28,1))
        pred.append(predictions.argmax())

## print confusion matrix for own prediction
mlp_cm = conf_matrix(andrius_labels, pred)    
    
    
    
