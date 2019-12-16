# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 09:53:35 2018

@author: Ankita
"""

import pandas as pd

import requests
import matplotlib.pyplot as plt
from skimage import io

photourl = pd.read_csv("C:/Users/ARUN/Desktop/datasets/red.csv")
photourlhead = photourl[10078:10078]
urlist = photourlhead.img_url.tolist()
pidlist = photourlhead.img_id.tolist()
img =[]

for i in range(1000):
    print(i)
    img_data = requests.get(urlist[i]).content
    with open('E:/ResearchFSU/Reddit/RedditImages/'+str(pidlist[i])+'.jpg', 'wb') as handler:
        handler.write(img_data)
        
#import pandas as pd
#mat = scipy.io.loadmat('C:/Users/ARUN/Downloads/viral_datasetV2.mat')
#
#M = mat['viral_dataset'] # array of structures
#df= pd.DataFrame(data=M[0,0:],    # values
#              # 1st column as index
#)
#df.to_csv("reddit.csv")

