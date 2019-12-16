# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:37:02 2018

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
import tensorflow as tf
import keras
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sklearn
from keras  import losses
import mmd
import tensorflow as tf
#import sklearn.metrics.pairwise.rbf_kernel 
sess = tf.InteractiveSession()


#kernel=utils.gaussian_kernel_matrix
a = np.array([[1, 1], [1, 2]])

b = np.array([[1, 1], [3, 0]])
c = np.array([[1],[2]])
f = np.linalg.norm(a - b)

#x_train = a
#x_train = np.concatenate((a, b, a, b),axis = 0  )
x_train = np.concatenate((a, b), axis = 0)
o1 = x_train[0:2]
o2 = x_train[1:2]
l3 = sklearn.metrics.pairwise.rbf_kernel(o1, Y=o2, gamma=None)
os = o1-o2
osq = np.mean(np.square(os))
o3 = x_train[2:3]
o4 = x_train[3:4]
os1 = o3-o4
osq1 = np.mean(np.square(os1))
#y_train = np.array([1,1])
y_train = np.array([[1, 1,1, 0]])
#y_train = np.concatenate((y_train, y_train),axis = 1 ) 

#fun = lambda x: np.square(x),1
#l3 = fun(one-two)


y_train = y_train.T

    ###################NN##############################################
# Loss function:
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        

def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))
  
    
def mmd_calculate(X,Y, kernel=mmd.pairwise_gaussian_kernel):
    cost = K.mean(kernel(X, X))
    cost += K.mean(kernel(Y, Y))
    cost -= 2 * K.mean(kernel(X, Y))
    return cost


def mean(x):
    return K.mean(x, axis=1, keepdims=True)

def output_of_lambda(input_shape):
    return (input_shape[0], 1)


def customLoss(out):
    def loss_new(y_true, y_pred):
        


# Some tensor we want to print the value of


# Add print operation
        
       
        loss1 = K.mean(K.square(y_pred-y_true), axis = -1)#//SUPERVISED loss
#        outis = tf.Print(out[0, 1, :, :], out[0, 1, :, :], message="This is a: ")
        
        out1 = out[0:2, :]
        out2 = out[2:4, :]
##        out1 = K.squeeze(out1, axis = 0)
#        batch_size, n_elems = out1.get_shape()
        unsup1 = y_pred[0:2]
        unsup2 = y_pred[2:4]
        l = mmd.smoothing
        loss3 = l(unsup1, unsup2, out1, out2)
        loss3 = loss3/2
        print("loss3.shape")
        print(loss3.shape)
#        loss3.eval()
#        
#       
#        print(out1.shape)
#        
#        
        
#        
#        loss2 = tf.reduce_mean(tf.square(out1- out2))
##        Kss = sklearn.metrics.pairwise.rbf_kernel(out1, Y=None, gamma=None)
##        Ktt = sklearn.metrics.pairwise.rbf_kernel(out2, Y=None, gamma=None)
##        Kst = sklearn.metrics.pairwise.rbf_kernel(out1, Y = out2, gamma = None)
#        
#        loss2 = K.mean(K.square(out1-out2))
        loss = loss2
        return loss
    return loss_new
    
epochs = 1

input_shape = (2,)

sess.close()
model = Sequential()

adel = optimizers.SGD(lr = 0.01)
model.add(Lambda(lambda x: x + 0, input_shape= input_shape, name = 'one'))
model.add(Lambda(mean, output_shape=output_of_lambda))

#model.add(Dense(1, activation='linear', name='predictions', kernel_initializer = 'zeros', bias = 'zeros'))

model.summary()
model.compile(loss=customLoss(model.get_layer("one").output),
              optimizer=adel,
              metrics=['accuracy'])

his1 = LossHistory()
history = model.fit(x_train, x_train,
          batch_size=4,
          epochs=epochs,
          verbose=2, callbacks = [his1]
       
          )

print(his1.losses)
pred = model.predict(x_train)
#out = model.get_layer("predictions1").output
#print_op = out.eval()
#x_l = model.get_layer("conv5").output

#MMD = K.mean(xx) - 2 * K.mean(xy) + K.mean(yy)
#        #return the square root of the MMD because it optimizes better
#        return K.sqrt(MMD); 
    