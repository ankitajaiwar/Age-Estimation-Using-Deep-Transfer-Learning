# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:06:18 2018

@author: Ankita
"""

import numpy as np
import pandas as pd

df = pd.read_csv("wiki.csv")
df["Age(in days)"] = df["Age(in days)"]/36865
df = df[df["Age(in days)"]>0] 
df = df[df["Age(in days)"]<1.10]
df.to_csv("wiki_processed.csv")