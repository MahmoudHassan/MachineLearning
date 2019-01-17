#Import numpy package for scientific computing
import numpy as np
#Used to create 2D graphs and plots by using python script
import matplotlib.pyplot as plt
#K-means clustering is one of the simplest and popular unsupervised machine learning algorithms
from sklearn.cluster import KMeans
#Pre-defined styles provided by Matplotlib
from matplotlib import style

#Pre-defined style called “ggplot”
style.use("ggplot")

#Inout data set
input_array=np.array([[3,2],[6,6],[2.6,3],[7,8],[3.5,5],[6,11]])


#instance from K-means Cluster
model=KMeans(n_clusters=2)

#fit model
model.fit(input_array)
#KMeans cluster centers
print("KMeans cluster centers is ",model.cluster_centers_)
#KMeans labels
print("KMeans labels is ",model.labels_)

#Draw predicted clusters
plt.scatter(input_array[:,0],input_array[:,1],c=model.labels_)

#Draw predicted cluster centers
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],marker='x',s=250,linewidths=5)
#Display all figures
plt.show()