# -*- coding: utf-8 -*-
"""
This Python script is used for classification of graphs which are memorized by testees.
In this script, classifiers are trained by EEG data within different frequency ranges, 
in different brain region pairs.
"""

import scipy.io as sio;
import numpy as np;
import pandas as pd;
import sklearn;
from sklearn import svm;
from time import perf_counter

######Data input######
data_path = data_path\\Neural_signals\\'

person1_os = sio.loadmat(data_path + 'Person1\\OSPerson1.mat')
person2_os = sio.loadmat(data_path + 'Person2\\OSPerson2.mat')
person3_os = sio.loadmat(data_path + 'Person3\\OSPerson3.mat')
person4_os = sio.loadmat(data_path + 'Person4\\OSPerson4.mat')
pair54 = sio.loadmat(data_path + 'Pair54.mat')
channame = sio.loadmat(data_path + 'ChanName.mat')
######################


######To set Parameters#####
##Brain region pairs
pairs = pair54['Pair54']
pair_symbols = pairs[:,0].astype('str')+pd.Series(['-']*len(pairs))+pairs[:,1].astype('str')

##Testees
people = [person1_os,person2_os,person3_os,person4_os]

##Frequency range of different waves
alpha_w = (16,20)  #frequency: 9-13 Hz
beta_w = (21,32)  #frequency: 14-30 Hz
theta_w = (4,15)  #frequency: 4-8 Hz
gamma_w = (33,51) #frequency: 30-100Hz   ###not used in th homework
wave_dict = {alpha_w:'alpha',beta_w:'beta',\
             theta_w:'theta',gamma_w:'gamma'}

##Kernel ID
kernel_ids = ['rbf','linear','sigmoid']
############################
    
    
#####To construct the classifier#####   
def lab_convert(arr):
    mem_label = list();
    for num in arr:
        if num <= 10:
            mem_label.append(1)
        else:
            mem_label.append(0)
    return np.array(mem_label)

def class_construct(wave,seed_num,kernel_id,multi_label):
    record = pd.DataFrame(np.zeros((len(pairs),seed_num)),index = pair_symbols,\
                          columns = np.arange(1,seed_num + 1))
    for i in range(len(pairs)):
        for j in range(4):   #four people
            person = people[j];
            os = person['OS'];
            fos = person['fOS'];
            track = person['Track'][0,:];
            for k in range(len(track)):  #To distinguish the graphs/trials
                mat = os[:,:,k,i];
                label_array = np.full((len(mat),1),track[k]);
                ###Old version
                ###mat = np.c_[mat,label_array];
                if k == 0:  ###the first matrix
                    merged_mat = mat;
                    merged_lab_arr = label_array
                else:
                    merged_mat = np.r_[merged_mat,mat]
                    merged_lab_arr = np.r_[merged_lab_arr,label_array]
                '''
                ###Old version
                if k == 0:  ###the first matrix
                    merged_mat = mat;
                else:
                    merged_mat = np.r_[merged_mat,mat]
                '''             
            '''
            ###Old version
            if multi_label == 'no':
                merged_mat[:,-1] = lab_convert(merged_mat[:,-1])
            else:
                pass
            '''
        #To split the data and labels
        ###Old version
        ###rawdata,labels=np.split(merged_mat,indices_or_sections=(mat.shape[1]-1,),axis=1)
        ###rawdata = rawdata[:,wave[0]:wave[1]+1]  #frequency splice
        rawdata = merged_mat[:,wave[0]:wave[1]+1]
        if multi_label == 'no':
            labels = lab_convert(merged_lab_arr);
        else:
            labels = merged_lab_arr
                    
        for seed in range(1,seed_num+1):
            train_set,test_set,train_labs,test_labs =sklearn.model_selection.train_test_split(rawdata,labels,\
                                                                                          random_state=seed,\
                                                                                          train_size=0.9,\
                                                                                          test_size=0.1)                                                                   
            #To train svm
            os_class=svm.SVC(C = 2,kernel= kernel_id,gamma = 10,decision_function_shape = 'ovr') 
            os_class.fit(train_set,train_labs.ravel())
            #4. To calculate the accurancy of svc
            ##record the results
            record.loc[pair_symbols[i],seed] = os_class.score(test_set,test_labs)
    
    #Output of the records
    record.to_csv('v5.record_{0}_{1}_{2}_{3}.txt'.format(kernel_id,wave_dict[wave],multi_label,seed_num), \
                  sep = '\t',index = True, header=True)

#####An example####
'''
for kernel in kernel_ids:
    for wave in wave_dict.keys():
        class_construct(wave,10,kernel,'no')
#wave,seed_num,kernel_id,multi_label
'''
start = perf_counter()
for wave in wave_dict.keys():
    class_construct(wave,500,'rbf','yes')
print('Time consumed: {}s'.format(perf_counter()-start))
