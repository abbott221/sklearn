from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
import pandas as pd
from sklearn import metrics



### LOAD AND PREPARE THE DATA

# Load dataset
df = read_csv("data/iris.csv")

# Prepare data
feature_cols = ['sepal_length','sepal_width','petal_length','petal_width']

X = df[feature_cols]


### RUN THE MODEL

# Configure and train our model: K-Means Clusterer
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(X)

# Predict cluster index for each sample
y_pred = kmeans.predict(X)



### EVALUATE THE MODEL

# Print the centers determined by the algorithm
print("Centers:\n" + str(kmeans.cluster_centers_) )

# Evaluate the result
labels = kmeans.labels_
print( "Silhouette Score: " + str(metrics.silhouette_score(X, labels, metric='euclidean')) )