__title__ = ''
__author__ = 'Sayo Makinwa'
__copyright__ = 'Copyright 2020, Thesis Project'
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1

#paths to file
feature_of_counts = "../processed_data/feature_vectors_counts.csv"

# Importing the dataset
dataset = pd.read_csv(feature_of_counts, index_col=0)
X = dataset.iloc[:,1:9].values
y = dataset.iloc[:, 9].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size = 0.3)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting RFC to the Training set
clf = RFC().fit(X_train, y_train)

# use our model to predict
y_pred = clf.predict(X_test)

#compute accuracy_score
accuracy = acc(y_test, y_pred)
print('accuracy', accuracy)

#compute precision score
precision_score = precision(y_test, y_pred, average='micro')
print('precision', precision_score)

#compute recall score
recall_score = recall(y_test, y_pred)
print('recall', recall_score)

#compute f1 score
f1_score = f1(y_test, y_pred)
print('f1', f1_score)