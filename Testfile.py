
# coding: utf-8

# In[1]:

#Plotting
get_ipython().magic('matplotlib inline')
#% matplotlib tk

#Imports
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from cvxopt import matrix, solvers
from sklearn.kernel_approximation import RBFSampler

#Mes imports 
os.chdir('/Users/tanguymarion/Desktop/MIT courses/Machine learning/Pb_set2/')
import Logistic_classification as LR_class



# In[4]:

# Load data
os.chdir('/Users/tanguymarion/Desktop/MIT courses/Machine learning/Pb_set2/hw2_resources/data')
name = '3'
# load data from train csv files
train = np.loadtxt('data'+name+'_train.csv')
X_tr = train[:,0:2]
y_tr = np.array(train[:,2:3])

# load validate data from csv files
validate = np.loadtxt('data'+name+'_validate.csv')
X_v = validate[:,0:2]
y_v = validate[:,2:3]

# plot training results on test set
test = np.loadtxt('data'+name+'_test.csv')
X_t = test[:,0:2]
y_t = np.array(test[:,2:3])



# In[5]:

#Logistique regression
#Logistic_classification(X_t, y_t, X_tr, y_tr, X_v, y_v, Plot, Display tuning, Tune, Lambda if no tuning, 
#                        reg if no tuning)

LR_class.Logistic_classification(X_t,y_t,X_tr,y_tr,X_v,y_v,True,True,True,1,'l2')


# In[30]:

#Linear_SVM
#Linear_SVM_classification(X_t, y_t, X_tr, y_tr, X_v, y_v, Plot, Display tuning, Tune, C if no tuning)
# Orange : support vectors positifs ; bleu : support vector négatifs

Lin_SVM_class.Linear_SVM_classification(X_t,y_t,X_tr,y_tr,X_v,y_v,True,False,False,1)


# In[6]:

#Kernalized_SVM
#Kernelized_SVM_classification(X_t,y_t,X_tr,y_tr,X_v,y_v,'rbf' or 'linear' kernel, plot,tune,C if no tuning,
#                             gamma if no tuning)

#Ker_SVM_class.Kernelized_SVM_classification(X_tr,y_tr,X_tr,y_tr,X_v,y_v,'rbf',True,False,100,20)


# In[5]:

# Linear Pegasos
# Lin_Pegasos_classification(X_t, y_t, X_tr, y_tr, X_v, y_v, max_epochs, plot, display, tune, lambda if no tuning)

#Lin_Peg_Class.Lin_Pegasos_classification(X_tr,y_tr,X_tr,y_tr,X_v,y_v,300,True,False,True,1)


# In[16]:

#Kernalized Pegasos
# Ker_Pegasos_classification(X_t, y_t, X_tr, y_tr, X_v, y_v, max_epochs, plot, lamb, gamma):

#Ker_Peg_Class.Ker_Pegasos_classification(X_tr,y_tr,X_tr,y_tr,X_v,y_v,500,False,0.0005,200)


# In[ ]:



