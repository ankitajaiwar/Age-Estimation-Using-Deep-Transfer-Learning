# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:54:42 2018

@author: Ankita
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:31:02 2018

@author: Ankita
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 15:02:45 2018

@author: Ankita
"""

import numpy as np
import pandas as pd
from skimage import io; io.use_plugin('matplotlib')
import matplotlib.pyplot as plt
import cv2
import scipy.io
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sklearn
from keras  import losses
import mmd
#import utils
import tensorflow as tf
#import sklearn.metrics.pairwise.rbf_kernel 
sess = tf.InteractiveSession()
#kernel=utils.gaussian_kernel_matrix
x = [[1,1], [1,1]]
y = [[0,1],[1,2]]

nt = 6940#flickr
ns = 9977#Reddit

nl = 1440#flickr
nu = 5500#flickr
n_test = 5500#flickr
#ns = 3480
#nl = 870
#nu = 2610
#nt = nl+nu
#ntest = 4000
n_train = 11424


df_male = pd.read_csv("../TLIP_FSU_Research/Datasets/Flickr_Dataset/flicker_new_dataset.csv")
df_female = pd.read_csv("../TLIP_FSU_Research/Datasets/Reddit_Dataset/red.csv")

df_male_labeled = df_male[0:nl]
df_male_unlabeled = df_male[nl:nl+nu]


img_rows = 128
img_cols = 128

#train_df = df.head(10000)
#final_df = np.empty((,6))
final_df = pd.DataFrame()
j = 0
k = 0
for i in range(0,714):
    
    k=i
    l =i
    if i+8 >ns:
        
        final_df = final_df.append(df_female[j:j+8])
        j = j+8
    else:
        final_df = final_df.append(df_female[i:i+8])
    if i+2 >nl:
        
        final_df = final_df.append(df_male_labeled[k:k+2])
        k = k+2
    else:
        final_df = final_df.append(df_male_labeled[i:i+2])
        
    if i+6>nu:
        
        final_df = final_df.append(df_male_unlabeled[l:l+6])
        l = l+6
    else:
        final_df = final_df.append(df_male_unlabeled[i:i+6])
        
final_df = final_df.reset_index(drop = True)
x_train = np.zeros((n_train, 128, 128,3))
y_train = np.zeros((n_train,1))
x_test = np.zeros((n_test, 128, 128,3))
y_test = np.zeros((n_test,1))
#foldertr = np.zeros((n_train,1))
#foldertr = foldertr.reshape(foldertr.shape[0],1)
#filetr = np.empty((n_train,1), dtype = object)
#filetr = filetr.reshape(filetr.shape[0],1)
#folderte = np.zeros((n_test,1))
#folderte = folderte.reshape(folderte.shape[0],1)
#filete = np.empty((n_test,1), dtype = object)
#filete = filete.reshape(filete.shape[0],1)
#gendertr = np.empty((n_train,1), dtype = object)
#gendertr = gendertr.reshape(gendertr.shape[0],1)
#genderte = np.empty((n_test,1), dtype = object)
#genderte = genderte.reshape(genderte.shape[0],1)
i =0
for index, row in final_df.iterrows():
    
#    print(index)
    
#    folder = str(row['folder'])
    image_name = str(row['img_id'])
    len_image_name = len(image_name)
    print(image_name)
    
    
    
#    image_name = image_name[:-2]
    try:
        
        image = io.imread('../TLIP_FSU_Research/Datasets/Reddit_Dataset/Images/'+image_name+'.jpg')
    except FileNotFoundError:
        print(index)
#        df_male = final_df.drop([index])
        i = i+1
        continue
    
    face = cv2.resize(image, (128, 128))
    
    if face.ndim <3:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    x,y,z = face.shape
    if z >3:
        face = cv2.cvtColor(face, cv2.COLOR_RGBA2BGR)
    
    x_train[i] = face
    y_train[i] = row['log_norm_score']
    
    print(i)
    if i == n_train-1:
        break
    i = i+1
x_train /= 255
#scipy.io.savemat('x_train_wiki.mat', mdict={'x_train': x_train, 'y_train': y_train, 'folder': foldertr, 'file': filetr})
print("Test started")
test_df = df_male[nt:nt+n_test]
i = 0
for index, row in test_df.iterrows():
    print(index)
    
#    folder = str(row['folder'])
    image_name = str(row['img_id'])
    len_image_name = len(image_name)
    
    
    if len_image_name >= 12:
    
        image_name = image_name[:-2]
    try:
        
        image = io.imread('../TLIP_FSU_Research/Datasets/Flickr_Dataset/Images/'+image_name+'.jpg')
    except FileNotFoundError:
        print("dude")
        i = i+1
        continue
#        test_df = test_df.drop([index])
    
    
    face = cv2.resize(image, (128, 128))
    if face.ndim <3:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    x,y,z = face.shape
    if z >3:
        face = cv2.cvtColor(face, cv2.COLOR_RGBA2BGR)
    
    
    x_test[i] = face
    y_test[i] = row['log_norm_score']
    
    print(i)
    if i == n_test-1:
        break
    i = i+1
x_test /= 255

#scipy.io.savemat('x_test_wiki.mat', mdict={'x_test': x_test, 'y_test': y_test, 'folder': folderte, 'file': filete})    

#np.savetxt("x_test_wiki.csv", x_test)
#np.savetxt("y_test_wiki.csv",y_test)
#np.savetxt("folder_test_wiki.csv", foldert)
#np.savetxt("file_file_wiki.csv",filet)    
    ###################NN##############################################
# Loss function:
def mmd_calculate(X,Y ):
    
    loss = K.mean(mmd.pairwise_gaussian_kernel(X, X))+K.mean(mmd.pairwise_gaussian_kernel(Y, Y))-2*K.mean(mmd.pairwise_gaussian_kernel(X, Y))
     
    return loss
def customLoss(out):
    def loss_new(y_true, y_pred):
        loss1 = K.mean(K.square(y_pred-y_true), axis = -1)#//SUPERVISED loss
#        outis = tf.Print(out[0, 1, :, :], out[0, 1, :, :], message="This is a: ")
        out1 = out[0:8,  :]
        print(out1.shape)
        
        out2 = out[8:16, :]
        print(out2.shape)
        
        unsup1 = y_pred[0:8]
        unsup2 = y_pred[8:16]
        l = mmd.smoothing
#        loss3 = l(unsup1, unsup2, out1, out2)
        
        loss2 = mmd_calculate(out1, out2)
        loss = loss1+loss2
        return loss
    return loss_new
    
epochs = 1000
    
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols)
#    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
#    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 3)

  
model = Sequential()

adel = optimizers.Adam(lr = 0.00001)

model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', padding='same', name='conv1'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool1'))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool2'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4'))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool3'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool4'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='conv6'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool5'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv7'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='pool6'))
#model.add(BatchNormalization())

model.add(Flatten(name='flatten'))

model.add(Dense(1, activation='sigmoid', name='predictions'))

model.summary()

model.compile(loss=customLoss(model.get_layer("flatten").output),
              optimizer=adel,
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=16,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test)
          )
#x_l = model.get_layer("conv5").output
score = model.evaluate(x_test, y_test, verbose=False)
prediction = model.predict(x_test)
fig = plt.figure()
plt.title("Loss vs Epoch for Train and Test without Random Labels")
plt.xlabel("Epochs")
plt.ylabel(" Loss")
plt.plot(history.history['loss'], label = "Training Loss", color = 'g')
plt.plot(history.history['val_loss'], label = "Test Loss", color = 'r')
plt.legend()
plt.grid(True)
plt.show()



    