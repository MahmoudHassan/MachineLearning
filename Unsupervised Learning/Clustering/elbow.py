import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
# Importing the dataset
dataset= pd.read_csv('export_dataframe.csv')
Sum_of_squared_distances = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(dataset)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
