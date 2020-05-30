"""
Created on Sat Nov 10 21:10:09 2018

@author: Sayo Makinwa
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1

# paths to fill
feature_of_counts = "/home/lechu/Documents/UL/2020-1/PoC/frameworks/clasificacion/drebin-malware-analysis_sayomakinwa" \
                    "/processed_data/feature_vectors_counts.csv"
# feature_of_counts = "/home/lechu/Documents/UL/2020-1/PoC/frameworks/clasificacion/drebin-malware-analysis_sayomakinwa/processed_data/feature_vectors_counts.csv"

# Importing the dataset
dataset = pd.read_csv(feature_of_counts, index_col=0)

# cuando se usa dataset de Drebin
# X = dataset.iloc[:, 1:9].values
# y = dataset.iloc[:, 9].values

# cuando se usa otro dataset diferente al de Drebin
# X: 11 tipos de atributo (variables independientes)
X = dataset.iloc[:, 1:12].values
# y: etiqueta de la aplicación (0: benigna, 1: malware) (variable target)
y = dataset.iloc[:, 12].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size=0.3, random_state=0)
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

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix', '\n', cm)


# compute accuracy_score
accuracy = acc(y_test, y_pred)
print('Accuracy: ', accuracy, '-->', format(accuracy, ".2%"))

# compute precision score
precision_score = precision(y_test, y_pred, average='micro')
print('Precision: ', precision_score, '-->', format(precision_score, ".2%"))

# compute recall score
recall_score = recall(y_test, y_pred)
print('Recall: ', recall_score, '-->', format(recall_score, ".2%"))

# compute f1 score
f1_score = f1(y_test, y_pred)
print('F1 Score: ', f1_score, '-->', format(f1_score, ".2%"))

import seaborn as sn
import matplotlib.pyplot as plt
# from pandas_ml import ConfusionMatrix

data = {"y_actual": y_test, "y_predicted": y_pred}
# print(data)
df = pd.DataFrame(data, columns=['y_actual','y_predicted'])
confusion_matrix = pd.crosstab(df['y_actual'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'],margins=True)
# print(df)
# print(confusion_matrix)
#Confusion_Matrix = ConfusionMatrix(df['y_actual'], df['y_predicted'])
#Confusion_Matrix.print_stats()

sn.heatmap(confusion_matrix, annot=True,fmt='g', cbar=True)
plt.show()

print(y)
print(y.astype(int))

