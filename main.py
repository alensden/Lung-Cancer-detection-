# this programn helps predicting lung cancer
# using 3 algos
# DATASET from https://data.world/alensden/lung-cancer-prediction-model-nn/workspace/file?agentid=sta427ceyin&datasetid=survey-lung-cancer&filename=survey+lung+cancer.csv
import pandas as pd
import numpy as np
data_set = pd.read_csv("survey lung cancer.csv")
print("DISPLAYING THE FIRST SEVEN ENTRIES")
print(data_set.head(7))
print(data_set.shape)
# CHECKING FOR EMPTY ENTRIES
print("\n\n")
empty_entries = data_set.isna().sum()
print("Total empty entries (NaN,Na)")
print(empty_entries)
print("\n\n")
#TO FIND ALL DATA TYPES
print(data_set.dtypes)
#CONVERTING STRING TO NUMBER ARRAY
# Y SET
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
label_encoder_Gender = LabelEncoder()

#CHANGING THE OBJECT FORM STRING TO BINARY ENCODING
old_dataset_Y = data_set.iloc[:,-1].values
#.values to conert to array
print(old_dataset_Y)
labelencoder_Y = labelencoder_Y.fit_transform(data_set.iloc[:,-1].values)
label_encoder_Gender = label_encoder_Gender.fit_transform(data_set.iloc[:,0].values)
data_set.iloc[:,-1] = labelencoder_Y
data_set.iloc[:,0] = label_encoder_Gender
print("New encoded data")
print(labelencoder_Y)
# 1 == YES && 0 == NO
# CORELATION IN DATA
corelation = data_set.iloc[:,0:17].corr()
print(corelation)

# SPLITTING DATA

X = data_set.iloc[:,0:15].values
Y = data_set.iloc[:,-1].values

#DATA SPLITTING INTO TEST AND TRAIN
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print("BEFORE SCALING")
print(X_train)

print("\n\n")

# Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#WE ONLY SCALE X TEST SET AS Y IS ONLY BINARY ENCODING
X_test = sc.fit_transform(X_test)
X_train = sc.fit_transform(X_train)
print("AFTER SCALING")
print(X_train)

# MACHINE LEARNING MODELS
# WE USE THREE ALOGS LOGISTIC REGRESSION DECISIONTREE FOREST CLASSIFIER
def models(X_train,Y_train):
    # LOGISTIC REGRESSION
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state=0)
    log.fit(X_train,Y_train)

    # DECISION TREE
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy',random_state=0)
    tree.fit(X_train,Y_train)

    # RANDOM FOREST CLASSIFIER
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    forest.fit(X_train,Y_train)

    print("[0] LOGISTIC REGRESSION ACCURACY : ",log.score(X_train,Y_train))
    print("[1] DECISION TREE ACCURACY : ",tree.score(X_train,Y_train))
    print("[2] RANDOM FOREST CLASSIFIER ACCURACY : ",forest.score(X_train,Y_train))

    return log,tree,forest
model = models(X_train,Y_train)

# TESTING ON CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
for i in range(len(model)):
    print("MODEL ",i)
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
    print(cm)
    # ACCURACY OF ALGORITHM DURING TESTING
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    print("TESTING ACCURACY = ", accuracy)
    print("\n\n")
# PREDICTING
pred1 = model[0].predict(X_test)
print("LOGISTIC REGRESSION")
print(pred1)
print("\n\n")
pred2 = model[1].predict(X_test)
print("DECISION TREE")
print(pred2)
print("\n\n")
pred3 = model[2].predict(X_test)
print("RANDON CLASSIFIER")
print(pred3)
print("\n\n")
print("ACTUAL DATA")
print(Y_test)
# Visualising All The Data
print(len(pred1))
#68 positive cases total corect data
#70 positive according to random forest algo ( 2 error cases )
