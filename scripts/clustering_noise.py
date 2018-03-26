from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
import pandas as pd
from sklearn import metrics



### LOAD AND PREPARE THE DATA

# Load dataset
df = read_csv("data/noisy_blobs.csv")

# Prepare data
feature_cols = ['a', 'b', 'c', 'd', 'e']

X = df[feature_cols]


### RUN THE MODEL

# Configure and train our model: K-Means Clusterer
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(X)

# Predict cluster index for each sample
y_pred = kmeans.predict(X)

################################################################################

# VISUALIZATION CODE

# Configure plotting
colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
colors = np.hstack([colors] * 20)

# Plot clustering result
plt.scatter(X.a, X.b, color=colors[y_pred].tolist(), s=10)
plt.show()
plt.scatter(X.c, X.d, color=colors[y_pred].tolist(), s=10)
plt.show()


################################################################################


from sklearn import decomposition
pca = decomposition.PCA(n_components=5)
pca.fit(X)

pcaX = pca.transform(X)

df = pd.DataFrame(pcaX)

plt.scatter(df[0], df[1], s=10)
plt.show()

plt.scatter(X[2], X[3], s=10)
plt.show()


