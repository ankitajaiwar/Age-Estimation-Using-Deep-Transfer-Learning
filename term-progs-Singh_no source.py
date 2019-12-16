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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sklearn
from keras  import losses
import term_progs_Singh_losses_mmd
#import utils
import tensorflow as tf
#import sklearn.metrics.pairwise.rbf_kernel 
sess = tf.InteractiveSession()
#kernel=utils.gaussian_kernel_matrix
x = [[1,1], [1,1]]
y = [[0,1],[1,2]]


ns = 3480
nl = 870
nu = 2610
nt = nl+nu
ntest = 4000
n_train = 870
n_test= 4000
df_mal = pd.read_csv("final_male_list_cleaned.csv")
df_male = df_mal[df_mal.folder<39]
df_male_labeled = df_male[0:nl]
df_male_unlabeled = df_male[nl:nl+nu]
df_fem = pd.read_csv("final_female_list_cleaned.csv")
df_female_x = df_fem[df_fem.folder<39]
df_female = df_female_x[0:ns]

img_rows = 128
img_cols = 128
final_df = df_male_labeled
#train_df = df.head(10000)
#final_df = np.empty((,6))
#final_df = pd.DataFrame()
#for i in range(0,435):
#    if i+8 <ns:
#        final_df = final_df.append(df_female[i:i+8])
#    
#    final_df = final_df.append(df_male_labeled[i:i+2])
#    final_df = final_df.append(df_male_unlabeled[i:i+6])
#    

x_train = np.zeros((n_train, 128, 128,3))
y_train = np.zeros((n_train,1))
x_test = np.zeros((n_test, 128, 128,3))
y_test = np.zeros((n_test,1))
foldertr = np.zeros((n_train,1))
foldertr = foldertr.reshape(foldertr.shape[0],1)
filetr = np.empty((n_train,1), dtype = object)
filetr = filetr.reshape(filetr.shape[0],1)
folderte = np.zeros((n_test,1))
folderte = folderte.reshape(folderte.shape[0],1)
filete = np.empty((n_test,1), dtype = object)
filete = filete.reshape(filete.shape[0],1)
gendertr = np.empty((n_train,1), dtype = object)
gendertr = gendertr.reshape(gendertr.shape[0],1)
genderte = np.empty((n_test,1), dtype = object)
genderte = genderte.reshape(genderte.shape[0],1)
i =0
for index, row in final_df.iterrows():
    
#    print(index)
    
    folder = str(row['folder'])
    image_name = str(row['file'])
    try:
        
        image = io.imread('../Age Wiki Clean/'+folder+'/'+image_name)
    except FileNotFoundError:
        print(index)
        df_male = final_df.drop([index])
        continue
    
    face = cv2.resize(image, (128, 128))
    
    if face.ndim <3:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    x,y,z = face.shape
    if z >3:
        face = cv2.cvtColor(face, cv2.COLOR_RGBA2BGR)
    
    x_train[i] = face
    y_train[i] = row['age']
    foldertr[i] = row['folder']
    filetr[i] = row['file']
    gendertr[i] = row['Gender']
    print(i)
    if i == n_train-1:
        break
    i = i+1
x_train /= 255
scipy.io.savemat('x_train_wiki.mat', mdict={'x_train': x_train, 'y_train': y_train, 'folder': foldertr, 'file': filetr})

test_df = df_male[nt:nt+ntest]
i = 0
for index, row in test_df.iterrows():
    print(index)
    
    folder = str(row['folder'])
    image_name = str(row['file'])
    try:
        
        image = io.imread('../Age Wiki Clean/'+folder+'/'+image_name)
    except FileNotFoundError:
        test_df = test_df.drop([index])
    
    
    face = cv2.resize(image, (128, 128))
    if face.ndim <3:
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    x,y,z = face.shape
    if z >3:
        iface = cv2.cvtColor(face, cv2.COLOR_RGBA2BGR)
    
    
    x_test[i] = face
    y_test[i] = row['age']
    folderte[i] = row['folder']
    filete[i] = row['file']
    genderte[i] = row['Gender']
    print(i)
    if i == n_test-1:
        break
    i = i+1
x_test /= 255

scipy.io.savemat('x_test_wiki.mat', mdict={'x_test': x_test, 'y_test': y_test, 'folder': folderte, 'file': filete})    

#np.savetxt("x_test_wiki.csv", x_test)
#np.savetxt("y_test_wiki.csv",y_test)
#np.savetxt("folder_test_wiki.csv", foldert)
#np.savetxt("file_file_wiki.csv",filet)    
    ###################NN##############################################
# Loss function:
def mmd_calculate(X,Y ):
    
    loss = K.mean(term_progs_Singh_losses_mmd.pairwise_gaussian_kernel(X, X))+K.mean(term_progs_Singh_losses_mmd.pairwise_gaussian_kernel(Y, Y))-2*K.mean(term_progs_Singh_losses_mmd.pairwise_gaussian_kernel(X, Y))
     
    return loss
def customLoss(out):
    def loss_new(y_true, y_pred):
        loss1 = K.mean(K.square(y_pred-y_true), axis = -1)#//SUPERVISED loss
#        outis = tf.Print(out[0, 1, :, :], out[0, 1, :, :], message="This is a: ")
#        out1 = out[0:8,  :]
#        print(out1.shape)
#        
#        out2 = out[8:16, :]
#        print(out2.shape)
#        
#        unsup1 = y_pred[0:8]
#        unsup2 = y_pred[8:16]
#        l = mmd.smoothing
#        loss3 = l(unsup1, unsup2, out1, out2)
#        
#        loss2 = mmd_calculate(out1, out2)
        loss = loss1
        return loss
    return loss_new
    
epochs = 1
    
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



plt.show()

square_error = mean_squared_error(y_test, prediction) 
absolute_error = mean_absolute_error(y_test, prediction) 
    