#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 15:23:03 2020

@author: jay
"""

import pickle
import pandas as pd

with open("./list_attendance1.pkl", 'rb') as file:
    list_attendance = pickle.load(file)

    
df = pd.read_csv("class_records.csv")
#df.values[0][2]=nana
attendace=df['attendance']

l2=[]
for name in list_attendance:
    temp=name.split("_")
    l2.append(temp[0])
    
Total_Names=df['Name']

for att in range(len(attendace)):
    if df["Name"][att] in l2:
        df['attendance'][att]='yes'
    else:
        df['attendance'][att]='no'

df.to_csv("class_records.csv", index=False)