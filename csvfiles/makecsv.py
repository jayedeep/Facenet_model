#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 11:48:13 2020

@author: jay
"""
import numpy as np
import csv

data = np.load('./5-celebrity-faces-embeddings.npz')
trainnames = data['arr_1']
all_names=list(set(trainnames))


rows=[]
for names in all_names:
    curr=names.split("_")
    rows.append(curr) 
    
fields = ['Name','enrollment','attendance']
  
filename = "class_records.csv"

with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
    # writing the fields  
    csvwriter.writerow(fields)  
        
    # writing the data rows  
    csvwriter.writerows(rows) 
    
    
with open("pickle_model.pkl", 'rb') as file:
    model = pickle.load(file)