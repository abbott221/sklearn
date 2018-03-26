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
just_this = df[["student"]]
just_this_encoded = pd.get_dummies(just_this)
new_df=pd.concat([df,just_this_encoded],axis=1)
df = new_df



df.default.replace('Yes',1,inplace=True)
df.default.replace('No',0,inplace=True)



feature_cols = ["student_No", "student_Yes", "balance", "income"]
target_cols = ["default"]



# Create training and test sets
df_train, df_test = train_test_split(df, test_size=0.1)



# THE DOWNSAMPLING
df_train_defaulted = df_train[df_train.default == 1]
df_train_didnt = df_train[df_train.default == 0]
df_train_didnt = df_train_didnt.sample(n=600)
sampled = pd.concat([df_train_defaulted, df_train_didnt])



X_train = sampled[feature_cols]
Y_train = np.ravel(sampled[target_cols])
X_test = df_test[feature_cols]
Y_test = np.ravel(df_test[target_cols])


# NORMALIZE THE DATA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)





### RUN THE MODEL

# Configure and train our model: Support Vector Machine

C_range = [0.25, 0.5, 1.0, 2.0, 4.0]
gamma_range = [0.25, 0.5, 1.0, 2.0, 4.0]



accuracy_lists = []
recall_lists = []
roc_auc_lists = []


from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


for C in C_range:
    accuracy = []
    recall = []
    roc_auc = []
    for gamma in gamma_range:
        print("C: " + str(C))
        print("gamma: " + str(gamma))

        clf = svm.SVC(kernel='rbf', C=C, gamma=gamma)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        acc = accuracy_score(Y_test, Y_pred)
        rec = recall_score(Y_test, Y_pred, average=None)
        auc = roc_auc_score(Y_test, Y_pred)
        print(acc)
        print(rec)
        print(auc)
        print()

        accuracy.append(acc)
        recall.append(rec)
        roc_auc.append(auc)
    accuracy_lists.append(accuracy)
    recall_lists.append(recall)
    roc_auc_lists.append(roc_auc)



print("\nACCURACY")

for the_list in accuracy_lists:
    str_builder = ""
    for x in the_list:
        str_builder += str(x) + "\t"
    print(str_builder)


print("\nRECALL")

for the_list in recall_lists:
    str_builder = ""
    for x in the_list:
        str_builder += "{0:.3f}".format(x[1]) + "\t"
    print(str_builder)


print("\nROC AUC")

for the_list in roc_auc_lists:
    str_builder = ""
    for x in the_list:
        str_builder += "{0:.3f}".format(x) + "\t"
    print(str_builder)


