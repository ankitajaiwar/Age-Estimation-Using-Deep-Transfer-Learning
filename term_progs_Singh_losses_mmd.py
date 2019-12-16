# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:24:33 2018

@author: Ankita
"""


import tensorflow as tf
from keras import backend as K




def pairwise_gaussian_kernel(x, y):
 
  
    n1 = x.get_shape().as_list()[1]
    n2 = y.get_shape().as_list()[1]
    if n1 is None:
        g = 10000
    else:
        g = n1
    gamma = -1.0 /float( g)
    norm = lambda x: K.sum(K.square(x), 1)
    dist =  norm(K.expand_dims(x, 2) - K.transpose(y))
    
    return K.exp(tf.scalar_mul(gamma,dist))
    
  


def smoothing(x,y, out1, out2):
   
    k = pairwise_gaussian_kernel(out1, out2)
    n1 = x.get_shape().as_list()[0]
    n2 = y.get_shape().as_list()[0]
    if n1 is None:
        n = 64
    else:
        n = n1*n2
    k = K.reshape(k, (n,1))
    func1 = lambda x: tf.subtract(x,y)
    f1 = tf.map_fn(func1,x)
    print(f1.shape)
    
#    f2 = K.sum(tf.multiply(k,K.reshape(K.square(f1), (n,1))))
    f2 = K.sum(tf.multiply(k,K.reshape(K.square(f1), (n,1))))
    return f2/2
    