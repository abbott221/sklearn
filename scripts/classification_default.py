from pandas import read_csv
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd



### LOAD AND PREPARE THE DATA

# Load dataset
df = read_csv("data/default.csv")

# Prepare data

# METHOD 1
just_this = df[["student"]]
just_this_encoded = pd.get_dummies(just_this)
new_df=pd.concat([df,just_this_encoded],axis=1)
df = new_df



# METHOD 2
df.default.replace('Yes',1,inplace=True)
df.default.replace('No',0,inplace=True)



feature_cols = ["student_No", "student_Yes", "balance", "income"]
target_cols = ["default"]



# Create training and test sets
df_train, df_test = train_test_split(df, test_size=0.1)

X_train = df_train[feature_cols]
Y_train = np.ravel(df_train[target_cols])
X_test = df_test[feature_cols]
Y_test = np.ravel(df_test[target_cols])


### RUN THE MODEL

# Configure and train our model: Stochastic Gradient Descent Classifier
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = clf.fit(X_train, Y_train)

# Predict remaining values
Y_pred = clf.predict(X_test)


### EVALUATE THE MODEL

# Evaluate the result
print(confusion_matrix(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))


from sklearn.metrics import recall_score
print(recall_score(Y_test, Y_pred, average=None))


from sklearn.metrics import roc_auc_score
print(roc_auc_score(Y_test, Y_pred))


