__title__ = ''
__author__ = 'Claudio Mori'
__copyright__ = 'Copyright 2020, Thesis Project'


import pandas as pd
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1


def separe_standarize_data(X, y):
    # Separación del conjunto de datos en datos de entrenamiento y datos de evaluacion. 30% para datos de evaluación y
    # 70% para datos de entrenamiento
    from sklearn.model_selection import train_test_split

    # X_train, X_test, y_train, y_test = train_test_split(X, y.astype(int), test_size = 0.3)
    train_features, test_features, train_labelapp, test_labelapp = \
        train_test_split(X, y.astype(int), test_size=0.3, train_size=0.7, random_state=15)

    '####################################### Estandarizar datos de 11 features ########################################'
    # Importar herramienta de estandarizador
    from sklearn.preprocessing import StandardScaler

    # Crear objeto estandarizador
    sc = StandardScaler()
    # Se va a estandarizar los datos por:
    # Reducir la variabilidad entre las frecuencias de aparición de cada feature que aparece en las aplicaciones
    # Se ajusta los datos de entrenamiento y transforma los datos de entrenamiento.
    train_features = sc.fit_transform(train_features)
    # Se transforman los datos de prueba
    test_features = sc.transform(test_features)

    return train_features, test_features, train_labelapp, test_labelapp


def obtain_metrics(test_labelapp, y_pred):
    '############################################## Calculo de metricas ###############################################'
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(test_labelapp, y_pred)
    print('Confusion Matrix', '\n', cm)

    # compute accuracy_score
    accuracy = acc(test_labelapp, y_pred)
    print('Accuracy: ', accuracy, '-->', format(accuracy, ".2%"))

    # calcular error
    error_rate = 1 - accuracy
    print('Error Rate: ', error_rate, '-->', format(error_rate, '.2%'))

    # compute precision score
    precision_score = precision(test_labelapp, y_pred, average='micro')
    print('Precision: ', precision_score, '-->', format(precision_score, ".2%"))

    # compute recall score
    recall_score = recall(test_labelapp, y_pred)
    print('Recall: ', recall_score, '-->', format(recall_score, ".2%"))

    # compute f1 score
    f1_score = f1(test_labelapp, y_pred)
    print('F1 Score: ', f1_score, '-->', format(f1_score, ".2%"))

    return error_rate, accuracy, precision_score, recall_score


def plot_confussion_matrix(nombre_modelo, test_labelapp, y_pred):
    # Draw Confussion Matrix using Heatmap
    import seaborn as sn
    import matplotlib.pyplot as plt

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

    sn.heatmap(confusion_matrix, annot=True, fmt='g', cbar=True, cmap="YlGnBu", xticklabels=True, yticklabels=True)
    plt.savefig('confussion-matrix-'+nombre_modelo+'.png')
    plt.show()
