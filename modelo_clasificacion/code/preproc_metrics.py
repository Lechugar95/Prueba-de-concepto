__title__ = ''
__author__ = 'Claudio Mori'
__copyright__ = 'Copyright 2020, Thesis Project'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import seaborn as sns

sns.set(font_scale=3)
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix


# Función para separación de datos y su estandarización con StandardScaler
def separe_standarize_data(X, y):
    # Separación del conjunto de datos en datos de entrenamiento y datos de evaluacion. 30% para datos de evaluación y
    # 70% para datos de entrenamiento
    # X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size = 0.3)
    train_features, test_features, train_labelapp, test_labelapp = \
        train_test_split(X, y.astype(int), test_size=0.3, train_size=0.7, random_state=15)
    '####################################### Estandarizar datos de 11 features ########################################'
    # Crear objeto estandarizador
    sc = StandardScaler()
    # Se va a estandarizar los datos para:
    # Reducir la variabilidad entre las frecuencias de aparición de cada feature que aparece en las aplicaciones
    # Se ajusta los datos de entrenamiento y transforma los datos de entrenamiento.
    train_features = sc.fit_transform(train_features)
    # Se transforman los datos de prueba
    test_features = sc.fit_transform(test_features)

    return train_features, test_features, train_labelapp, test_labelapp


# Funcion para calcular metricas
def obtain_metrics(test_labelapp, y_pred):
    '############################################## Calculo de metricas ###############################################'
    # Dibujando la matriz de confusion
    cm = confusion_matrix(test_labelapp, y_pred)
    print('Confusion Matrix', '\n', cm)

    # Calculando la Exactitud (Accuracy)
    accuracy = acc(test_labelapp, y_pred)
    print('Accuracy: ', accuracy, '-->', format(accuracy, ".2%"))

    # Calculando el ratio de error (Error Rate)
    error_rate = 1 - accuracy
    print('Error Rate: ', error_rate, '-->', format(error_rate, '.2%'))

    # Calculando la Precision (Precision)
    precision_score = precision(test_labelapp, y_pred, ".2%")
    print('Precision: ', precision_score, '-->', format(precision_score, ".2%"))

    # Calculando la Sensibilidad (Recall)
    recall_score = recall(test_labelapp, y_pred)
    print('Recall: ', recall_score, '-->', format(recall_score, ".2%"))

    # Calculando el Valor F (F1 Score)
    f1_score = f1(test_labelapp, y_pred)
    print('F1 Score: ', f1_score, '-->', format(f1_score, ".2%"))

    return error_rate, accuracy, precision_score, recall_score, f1_score


# Funcion para dibujar matriz de confusion con heatmap de Seaborn
def plot_confussion_matrix(nombre_modelo, test_labelapp, y_pred):
    # from pandas_ml import ConfusionMatrix
    data = {"y_actual": test_labelapp, "y_predicted": y_pred}
    # print(data)
    df = pd.DataFrame(data, columns=['y_actual', 'y_predicted'])
    confusion_matrix = pd.crosstab(df['y_actual'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'],
                                   margins=False)
    # print(df)
    # print(confusion_matrix)
    # Confusion_Matrix = ConfusionMatrix(df['y_actual'], df['y_predicted'])
    # Confusion_Matrix.print_stats()
    figsize(20, 18)
    sns.set(font_scale=2)
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cbar=True, cmap="YlGnBu", xticklabels=True, yticklabels=True)
    plt.savefig('../code/plots/confussion-matrix-'+nombre_modelo+'.png')
    plt.show()
