from pandas import read_csv
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn import cluster
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt



### LOAD AND PREPARE THE DATA

# Load dataset
df_train = read_csv("data/20groups_train.csv")
df_test = read_csv("data/20groups_test.csv")

# Prepare data
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.5, min_df=5, stop_words='english')
train_vectors = vectorizer.fit_transform(df_train['data'])
test_vectors = vectorizer.transform(df_test['data'])



X_train = train_vectors
Y_train = np.ravel(df_train['target'])
X_test = test_vectors
Y_test = np.ravel(df_test['target'])





### RUN THE MODEL

# Configure and train our model: K-Means Clusterer
kmeans = cluster.KMeans(n_clusters=20, max_iter=100, init='k-means++', n_init=1, verbose=1)
kmeans.fit(X_train)

# Predict cluster index for each sample
y_pred = kmeans.predict(X_train)



### EVALUATE THE MODEL

# Evaluate the result
labels = kmeans.labels_
print( "Silhouette Score: " + str(metrics.silhouette_score(X_train, labels, metric='euclidean')) )


