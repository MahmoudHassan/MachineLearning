# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 20:53:50 2019

@author: Mahmoud Mohamed Hassan
"""

#Import numpy package for scientific computing
import numpy as np
#Used to create 2D graphs and plots by using python script
import matplotlib.pyplot as plt
#Simple and efficient tools for data mining and data analysis
#import Support vector machines classifier
from sklearn import svm
#Pre-defined styles provided by Matplotlib
from matplotlib import style


#Pre-defined style called “ggplot”
style.use("ggplot")

#Inout data set
input_array=np.array([[3,2],[6,6],[2.6,3],[7,8],[3.5,5],[6,11]])
#output labels
output_array=np.array([0,1,0,1,0,1])

#Instance from Epsilon-Support Vector Regression classifier
#The kernel function can be any of the following:
#linear: .
#polynomial: .  is specified by keyword degree,  by coef0.
#rbf: .  is specified by keyword gamma, must be greater than 0.
#sigmoid (), where  is specified by coef0.
model=svm.SVR(kernel='linear')

#Then train the model by fitting it to the data 
model.fit(input_array,output_array)
#SVR predict [0.5,0.8]
print("SVR predict [0.5,0.8] is ",model.predict([[0.5,0.8]]))
#Draw predicted point [0.5,0.8]
plt.scatter(0.5,0.8,c='r')
#SVR predict [8.5,10]
print("SVR predict [8.5,10] is ",model.predict([[8.5,10]]))
#Draw predicted point [8.5,10]
plt.scatter(8.5,10,c='r')
#Draw input and output array 
plt.scatter(input_array[:,0],input_array[:,1],c=output_array)
#Display all figures
plt.show()