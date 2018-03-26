from pandas import read_csv
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd



### LOAD AND PREPARE THE DATA

# Load dataset
df_train = read_csv("data/20groups_train.csv")
df_test = read_csv("data/20groups_test.csv")

# Prepare data
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(df_train['data'])
test_vectors = vectorizer.transform(df_test['data'])



X_train = train_vectors
Y_train = np.ravel(df_train['target'])
X_test = test_vectors
Y_test = np.ravel(df_test['target'])



# Configure and train our model: Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf = clf.fit(X_train, Y_train)

# Predict remaining values
Y_pred = clf.predict(X_test)


### EVALUATE THE MODEL

# Evaluate the result
print(confusion_matrix(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))


