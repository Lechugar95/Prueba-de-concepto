__title__ = ''
__author__ = 'Claudio Mori'
__copyright__ = 'Copyright 2020, Thesis Project'

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from preproc_metrics import *
from gs_metrics import *


def svc_model():
    pd.set_option('display.max_columns', None)

    # paths to fill
    feature_of_counts = "/home/lechu/Documents/GitHub/Prueba-de-concepto/seleccion_caracteristicas/processed_data" \
                        "/feature_vectors_counts.csv"

    # Importando el dataset en forma de archivo .csv. El archivo se llama feature_vectors_counts.csv
    # Tiene 12 colmumnas: 11 para mostrar la frecuencia de aparición de los 11 feature extraídos de la aplicacióny 1 para
    # la varialbe target representada con el label o etiqueta que indica si la aplicación es malware o benigna.
    dataset = pd.read_csv(feature_of_counts, index_col=0)

    print(
        '######################### MODELO DE CLASIFICACION 6: SVC USANDO 11 FEATURES ##########################')

    # X: 11 tipos de atributo (variables independientes)
    X = dataset.iloc[:, 1:12].values
    # y: etiqueta de la aplicación (0: benigna, 1: malware) (variable target)
    y = dataset.iloc[:, 12].values
    '################################################ Separacion de datos ################################################'
    '''
    # Separación del conjunto de datos en datos de entrenamiento y datos de evaluacion. 30% para datos de evaluación y
    # 70% para datos de entrenamiento
    from sklearn.model_selection import train_test_split
    train_features, test_features, train_labelapp, test_labelapp = train_test_split(X, y.astype(int), test_size=0.3, train_size=0.7, random_state=15)
    '''
    '######################################### Estandarizar datos de 11 features ##########################################'
    '''
    # Importar herramienta de estandarizador
    from sklearn.preprocessing import StandardScaler
    
    # Crear objeto estandarizador
    sc1 = StandardScaler()
    # Ajustar los datos de entrenamiento
    sc1.fit(train_features)
    # Se va a estandarizar los datos por dos razones:
    # (1) Reducir la variabilidad entre las frecuencias de aparición de cada feature que aparece en las aplicaciones
    # (2) Se va a aplicar PCA antes de la elaboración del modelo, por lo que es necesario estanrizar los features de la data
    # Se transforma los datos de entrenamiento
    train_features = sc1.transform(train_features)
    # Se ajusta los datos de entrenamiento y se transforma los datos de entrenamiento y de prueba
    # train_features = sc.fit_transform(train_features)
    test_features = sc1.transform(test_features)
    '''
    train_features, test_features, train_labelapp, test_labelapp = separe_standarize_data(X, y)
    '##################################################### Uso de PCA #####################################################'
    '''
    from sklearn.decomposition import PCA
    # Crear la instancia del modelo de PCA
    # .95 se refiere que se va a elegir el número mínimo de componentse principales al retener el 95% de la varianza
    # pca_featuresapps = PCA(n_components=10)
    pca_featuresapps = PCA(n_components=11, svd_solver='randomized', iterated_power=100, random_state=15)
    # ajustar PCA con el conjunto de entrenamiento de datos que fue estandarizado
    pca_featuresapps.fit(train_features)
    # Mostrar cantidad de componentes usados por PCA
    print("Cantidad de componentes PCA elegidos: ", pca_featuresapps.n_components)
    # Aplicar la transformación a ambos conjuntos de datos: entrenamiento y de evaluación
    train_features = pca_featuresapps.transform(train_features)
    test_features = pca_featuresapps.transform(test_features)
    
    import matplotlib.pyplot as plt
    percentage_variation = np.round(pca_featuresapps.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(percentage_variation)+1)]
    
    # Draw a scree plot and a PCA plot
    plt.bar(x=range(1, len(percentage_variation)+1), height=percentage_variation, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()
    '''
    '###################################### MODELO DE CLASIFICACION 1: SVC ######################################'
    # clf_svc = SVC(C=1.0, kernel='rbf', gamma=0.73, max_iter=1000, random_state=15)
    clf_svc = SVC(C=1.0, kernel='rbf', gamma=0.73, max_iter=1000, random_state=15)
    clf_svc.fit(train_features, train_labelapp)
    y_pred = clf_svc.predict(test_features)

    '###################################### TUNEO DE HYPERPARAMETROS ######################################'
    '''
    # Configurar parametros para la validacion cruzada
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    import sklearn.metrics as sk
    print(sorted(sk.SCORERS.keys()))
    scores = ['precision', 'recall', 'f1']
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf_svc1 = GridSearchCV(
            SVC(), tuned_parameters, scoring='%s_macro' % score
        )
        clf_svc1.fit(train_features, train_labelapp)
    
        print("Best parameters set found on development set:")
        print()
        print(clf_svc1.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf_svc1.cv_results_['mean_test_score']
        stds = clf_svc1.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf_svc1.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = test_labelapp, clf_svc1.predict(test_features)
        print(classification_report(y_true, y_pred))
        print()
    '''

    print(
        '################################### Resultados del modelo usando 11 features ###################################')
    '################################################ Calculo de metricas #################################################'
    '''
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(test_labelapp, y_pred)
    print('Confusion Matrix', '\n', cm)
    
    # compute accuracy_score
    accuracy = acc(test_labelapp, y_pred)
    print('Accuracy: ', accuracy, '-->', format(accuracy, ".2%"))
    
    # compute precision score
    precision_score = precision(test_labelapp, y_pred)
    print('Precision: ', precision_score, '-->', format(precision_score, ".2%"))
    
    # compute recall score
    recall_score = recall(test_labelapp, y_pred)
    print('Recall: ', recall_score, '-->', format(recall_score, ".2%"))
    
    # compute f1 score
    f1_score = f1(test_labelapp, y_pred)
    print('F1 Score: ', f1_score, '-->', format(f1_score, ".2%"))
    '''
    '######################################### Graficar la 1ra. Matriz de confusion #######################################'
    '''
    # Draw Confussion Matrix using Heatmap
    import seaborn as sn
    import matplotlib.pyplot as plt
    
    # from pandas_ml import ConfusionMatrix
    data1 = {"y_actual": test_labelapp, "y_predicted": y_pred}
    # print(data)
    df2 = pd.DataFrame(data1, columns=['y_actual', 'y_predicted'])
    confusion_matrix2 = pd.crosstab(df2['y_actual'], df2['y_predicted'], rownames=['Actual'], colnames=['Predicted'],
                                    margins=False)
    # print(df)
    # print(confusion_matrix)
    # Confusion_Matrix = ConfusionMatrix(df['y_actual'], df['y_predicted'])
    # Confusion_Matrix.print_stats()
    
    sn.heatmap(confusion_matrix2, annot=True, fmt='g', cbar=True, cmap="YlGnBu", xticklabels=True, yticklabels=True)
    plt.show()
    '''
    error_rate_svc, accuracy_svc, precision_svc, recall_svc = obtain_metrics(test_labelapp, y_pred)
    plot_confussion_matrix('svc', test_labelapp, y_pred)

    return error_rate_svc, accuracy_svc, precision_svc, recall_svc
