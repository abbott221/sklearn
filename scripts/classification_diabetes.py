from pandas import read_csv
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score



### LOAD AND PREPARE THE DATA

# Load dataset
df = read_csv("data/diabetes.csv")

# Prepare data
feature_cols = ["NumberPregnant", "GlucoseConcentration", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DPFunction", "Age"]
target_cols = ["Class"]

# Create training and test sets
df_train, df_test = train_test_split(df, test_size=0.1)

X_train = df_train[feature_cols]
Y_train = np.ravel(df_train[target_cols])
X_test = df_test[feature_cols]
Y_test = np.ravel(df_test[target_cols])


### RUN THE MODEL

# Configure and train our model: Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
clf = gnb.fit(X_train, Y_train)

# Predict remaining values
Y_pred = clf.predict(X_test)


### EVALUATE THE MODEL

# Evaluate the result
print(confusion_matrix(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))