# -*- coding: utf-8 -*-
"""
Created on Fri Jul 06 15:50:05 2018

@author: Ankita
"""

import pandas as pd

import requests
import matplotlib.pyplot as plt

photourl = pd.read_csv("../../Datasets/Flickr_Dataset/flicker_new_dataset.csv")
x = 9001
y =12450
#for i in range(1500):
    
print("from range "+str(x)+"to"+str(y))
photourlhead = photourl[x:y]
urlist = photourlhead.url_o.tolist()
pidlist = photourlhead.id.tolist()
img =[]

for i in range(3500):
    img_data = requests.get(urlist[i]).content
    with open('../../Datasets/Flickr_Dataset/more/'+str(pidlist[i])+'.jpg', 'wb') as handler:
        handler.write(img_data)
#    x = x+100
#    y = y+100
#    
#    if(x == 1001):
#        break
#    
   