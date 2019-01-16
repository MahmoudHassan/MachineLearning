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

#instance from Support vector machines classifier
model=svm.SVC(kernel='linear')

#fit model
model.fit(input_array,output_array)
#SVC predict [0.5,0.8]
print("SVC predict [0.5,0.8] is ",model.predict([[0.5,0.8]]))
#Draw predicted point [0.5,0.8]
plt.scatter(0.5,0.8,c='r')
#SVC predict [8.5,10]
print("SVC predict [8.5,10] is ",model.predict([[8.5,10]]))
#Draw predicted point [8.5,10]
plt.scatter(8.5,10,c='r')
#Draw input and output array 
plt.scatter(input_array[:,0],input_array[:,1],c=output_array)
plt.show()