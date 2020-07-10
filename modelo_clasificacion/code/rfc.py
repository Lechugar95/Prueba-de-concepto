__title__ = ''
__author__ = 'Claudio Mori'
__credits__ = 'Sayo Makinwa'
__copyright__ = 'Copyright 2020, Thesis Project'

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# importar script creado para reutilizar funciones separe_standarize_data, obtain_metrics, plot_confussion_matrix
from preproc_metrics import *
# importar script para reutilizar función evaluate con el modelo base, de Random Grid Search y Grid search.
# funcion: evaluate(nombre_modelo, modelo_construido, features_prueba, labels_prueba)
from gs_metrics import *

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 10

import seaborn as sns
sns.set(font_scale=2)

from pprint import pprint
from IPython.core.pylabtools import figsize
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# from sklearn.metrics import accuracy_score as acc
# from sklearn.metrics import precision_score as precision
# from sklearn.metrics import recall_score as recall
# from sklearn.metrics import f1_score as f1

pd.set_option('display.max_columns', None)

# paths to fill
feature_of_counts = "/home/lechu/Documents/GitHub/Prueba-de-concepto/seleccion_caracteristicas/processed_data" \
                    "/feature_vectors_counts.csv"

# Importando el dataset en forma de archivo .csv. El archivo se llama feature_vectors_counts.csv
# Tiene 12 colmumnas: 11 para mostrar la frecuencia de aparición de los 11 feature extraídos de la aplicacióny 1 para
# la varialbe target representada con el label o etiqueta que indica si la aplicación es malware o benigna.
dataset = pd.read_csv(feature_of_counts, index_col=0)

print(
    '####################### MODELO DE CLASIFICACION 1: RANDOM FOREST BASE USANDO 11 FEATURES ########################')

# X: 11 tipos de atributo (variables independientes)
X = dataset.iloc[:, 1:12].values

# y: etiqueta de la aplicación (0: benigna, 1: malware) (variable target)
y = dataset.iloc[:, 12].values

'################################################ Separacion de datos ################################################'
'''
# Separación del conjunto de datos en datos de entrenamiento y datos de evaluacion. 30% para datos de evaluación y
# 70% para datos de entrenamiento
from sklearn.model_selection import train_test_split

train_features, test_features, train_labelapp, test_labelapp = \
    train_test_split(X, y.astype(int), test_size=0.3, train_size=0.7, random_state=15)
'''
'######################################### Estandarizar datos de 11 features ##########################################'
'''
# Importar herramienta de estandarizador
from sklearn.preprocessing import StandardScaler

# Crear objeto estandarizador
sc1 = StandardScaler()
# Se va a estandarizar los datos por:
# Reducir la variabilidad entre las frecuencias de aparición de cada feature que aparece en las aplicaciones
# Se ajusta los datos de entrenamiento y transforma los datos de entrenamiento.
train_features = sc1.fit_transform(train_features)
# Se transforman los datos de prueba
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
'###################################### MODELO DE CLASIFICACION 1: RANDOM FOREST ######################################'
# Fitting RFC to the Training set
# clf_rfc_11f = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True, max_depth=10, random_state=15)
# clf_rfc_11f = RandomForestClassifier(n_estimators=300, criterion='gini', bootstrap=True, max_depth=30, random_state=15)
clf_rfc_11f = RandomForestClassifier(n_estimators=1000, criterion='gini',
                                     bootstrap=True, max_depth=100, random_state=15)
clf_rfc_11f.fit(train_features, train_labelapp)
y_pred = clf_rfc_11f.predict(test_features)

print(
    '################################### Resultados del modelo usando 11 features ###################################')
'################################################ Calculo de metricas #################################################'
'''
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labelapp, y_pred1)
print('Confusion Matrix', '\n', cm)

# compute accuracy_score
accuracy = acc(test_labelapp, y_pred1)
print('Accuracy: ', accuracy, '-->', format(accuracy, ".2%"))

# compute precision score
precision_score = precision(test_labelapp, y_pred1, average='micro')
print('Precision: ', precision_score, '-->', format(precision_score, ".2%"))

# compute recall score
recall_score = recall(test_labelapp, y_pred1)
print('Recall: ', recall_score, '-->', format(recall_score, ".2%"))

# compute f1 score
f1_score = f1(test_labelapp, y_pred1)
print('F1 Score: ', f1_score, '-->', format(f1_score, ".2%"))
print(
    '################################################################################################################')
print('\n')
'''
'########################################### Graficar la Matriz de confusion #########################################'
'''
# Draw Confussion Matrix using Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

# from pandas_ml import ConfusionMatrix
data1 = {"y_actual": test_labelapp, "y_predicted": y_pred1}
# print(data)
df2 = pd.DataFrame(data1, columns=['y_actual', 'y_predicted'])
confusion_matrix2 = pd.crosstab(df2['y_actual'], df2['y_predicted'], rownames=['Actual'], colnames=['Predicted'],
                                margins=False)
# print(df)
# print(confusion_matrix)
# Confusion_Matrix = ConfusionMatrix(df['y_actual'], df['y_predicted'])
# Confusion_Matrix.print_stats()

sns.heatmap(confusion_matrix2, annot=True, fmt='g', cbar=True, cmap="YlGnBu", xticklabels=True, yticklabels=True)
plt.show()
'''

error_rate_rfc11f, accuracy_rfc11f, precision_rfc11f, recall_rfc11f = obtain_metrics(test_labelapp, y_pred)

plot_confussion_matrix('rfc-11features', test_labelapp, y_pred)

print('################## MODELO DE CLASIFICACION 2: RANDOM FOREST USANDO FEATURES SELECCIONADOS ####################')
print(
    '################################## Finding Important Features in Scikit-learn ##################################')
clf = RandomForestClassifier(n_estimators=1000, criterion='gini', bootstrap=True, max_depth=100, random_state=15)
clf.fit(train_features, train_labelapp)

feature_names = list(dataset.iloc[:, 1:12].columns)
# print(feature_names)

# Calcular la importancia de cada uno de los 11 features
feature_importance = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
print(feature_importance)

'########################################## Feature Importance Graph Barplot ##########################################'
# Crear barras de features del grafico
sns.barplot(x=feature_importance, y=feature_importance.index, seed=15)
# Añadir etiquetas de los features al grafico
plt.xlabel('Feature Importance Score')
plt.ylabel('11 Apks Features')
plt.title("Visualizing Important Features")
# plt.legend()
plt.savefig('barras-importancia-11features.png')
plt.show()

'##################################### Generating the Model on Selected Features ######################################'
# Crear modelo en base a las características más relevantes

# ELEGIR UNO DE LOS CONJUNTOS DE FEATURES PARA USARLO EN EL ENTRENAMIENTO DEL MODELO
# low 4 features
X_4l = dataset[['proveedor_contenido', 'urls', 'uses-feature', 'receptor_mensajes']]
# low 6 features
X_6l = dataset[['proveedor_contenido', 'urls', 'uses-feature', 'receptor_mensajes', 'filtros_intent', 'servicios']]
# top 5 features
X_5 = dataset[['permisos_solicitados', 'api_restringidas', 'actividades', 'api_sospechosas', 'permisos_usados']]
# top 7 features
X_7 = dataset[['permisos_solicitados', 'api_restringidas', 'actividades', 'api_sospechosas', 'permisos_usados', 'servicios', 'filtros_intent']]
# top 8 features
X_8 = dataset[['permisos_solicitados', 'api_restringidas', 'actividades', 'api_sospechosas', 'permisos_usados',
             'servicios', 'filtros_intent', 'receptor_mensajes']]
# top 10 features
X_10 = dataset[['permisos_solicitados', 'api_restringidas', 'actividades', 'api_sospechosas', 'permisos_usados', 'servicios', 'filtros_intent', 'receptor_mensajes', 'urls', 'uses-feature']]
y = dataset.iloc[:, 12].values

'################################################ Separacion de datos ################################################'
'''
train_features, test_features, train_labelapp, test_labelapp = \
    train_test_split(X, y.astype(int), test_size=0.3, train_size=0.7, random_state=15)
'''
'################################## Estandarizar datos de top features seleccionados ##################################'
'''
# Crear objeto estandarizador
sc2 = StandardScaler()
# Ajustar los datos de entrenamiento
sc2.fit(train_features)
# Se transforma los datos de entrenamiento
train_features = sc2.transform(train_features)
# Se ajusta los datos de entrenamiento y se transforma los datos de entrenamiento y de prueba
# train_features = sc.fit_transform(train_features)
test_features = sc2.transform(test_features)
'''
# Separacion de datos y Estandarizacion de los mismos
train_features_4l, test_features_4l, train_labelapp_4l, test_labelapp_4l = separe_standarize_data(X_4l, y)
train_features_6l, test_features_6l, train_labelapp_6l, test_labelapp_6l = separe_standarize_data(X_6l, y)
train_features_5, test_features_5, train_labelapp_5, test_labelapp_5 = separe_standarize_data(X_5, y)
train_features_7, test_features_7, train_labelapp_7, test_labelapp_7 = separe_standarize_data(X_7, y)
train_features_8, test_features_8, train_labelapp_8, test_labelapp_8 = separe_standarize_data(X_8, y)
train_features_10, test_features_10, train_labelapp_10, test_labelapp_10 = separe_standarize_data(X_10, y)

'###################################### MODELO DE CLASIFICACION 2: RANDOM FOREST ######################################'
# clf_rfc_topF = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True, max_depth=10, random_state=15)
# clf_rfc_topF = RandomForestClassifier(n_estimators=300, criterion='gini', bootstrap=True, max_depth=30, random_state=15)
# Modelo de clasificacion 2: Random Forest usando 10 top features
clf_rfc_low4f = RandomForestClassifier(n_estimators=1000, criterion='gini', bootstrap=True, max_depth=100, random_state=15)
clf_rfc_low6f = RandomForestClassifier(n_estimators=1000, criterion='gini', bootstrap=True, max_depth=100, random_state=15)
clf_rfc_top5f = RandomForestClassifier(n_estimators=1000, criterion='gini', bootstrap=True, max_depth=100, random_state=15)
clf_rfc_top7f = RandomForestClassifier(n_estimators=1000, criterion='gini', bootstrap=True, max_depth=100, random_state=15)
clf_rfc_top8f = RandomForestClassifier(n_estimators=1000, criterion='gini', bootstrap=True, max_depth=100, random_state=15)
clf_rfc_top10f = RandomForestClassifier(n_estimators=1000, criterion='gini', bootstrap=True, max_depth=100, random_state=15)

# Entrenamiento de modelo
clf_rfc_low4f.fit(train_features_4l, train_labelapp_4l)
clf_rfc_low6f.fit(train_features_6l, train_labelapp_6l)
clf_rfc_top5f.fit(train_features_5, train_labelapp_5)
clf_rfc_top7f.fit(train_features_7, train_labelapp_7)
clf_rfc_top8f.fit(train_features_8, train_labelapp_10)
clf_rfc_top10f.fit(train_features_10, train_labelapp_10)

# Obtención de perdicciones
y_pred_4l = clf_rfc_low4f.predict(test_features_4l)
y_pred_6l = clf_rfc_low6f.predict(test_features_6l)
y_pred_5 = clf_rfc_top5f.predict(test_features_5)
y_pred_7 = clf_rfc_top7f.predict(test_features_7)
y_pred_8 = clf_rfc_top8f.predict(test_features_8)
y_pred_10 = clf_rfc_top10f.predict(test_features_10)

print('######################## Resultados del modelo usando features con mayor importancia #########################')
'################################################ Calculo de metricas #################################################'
'''
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labelapp, y_pred2)
print('Confusion Matrix', '\n', cm)

# compute accuracy_score
accuracy = acc(test_labelapp, y_pred2)
print('Accuracy: ', accuracy, '-->', format(accuracy, ".2%"))

# compute precision score
precision_score = precision(test_labelapp, y_pred2, average='micro')
print('Precision: ', precision_score, '-->', format(precision_score, ".2%"))

# compute recall score
recall_score = recall(test_labelapp, y_pred2)
print('Recall: ', recall_score, '-->', format(recall_score, ".2%"))

# compute f1 score
f1_score = f1(test_labelapp, y_pred2)
print('F1 Score: ', f1_score, '-->', format(f1_score, ".2%"))
'''
'########################################### Graficar la Matriz de confusion #########################################'
'''
# Draw Confussion Matrix using Heatmap
import seaborn as sns
import matplotlib.pyplot as plt

# from pandas_ml import ConfusionMatrix
data2 = {"y_actual": test_labelapp, "y_predicted": y_pred2}
# print(data)
df2 = pd.DataFrame(data2, columns=['y_actual', 'y_predicted'])
confusion_matrix2 = pd.crosstab(df2['y_actual'], df2['y_predicted'], rownames=['Actual'], colnames=['Predicted'],
                                margins=False)
# print(df)
# print(confusion_matrix)
# Confusion_Matrix = ConfusionMatrix(df['y_actual'], df['y_predicted'])
# Confusion_Matrix.print_stats()

sns.heatmap(confusion_matrix2, annot=True, fmt='g', cbar=True, cmap="YlGnBu", xticklabels=True, yticklabels=True)
plt.show()
'''

print('######################## Resultados del modelo usando features con mayor importancia #########################')
print('########### Métricas obtenidas con 4lf ###########' )
error_rate_rfc4lf, accuracy_rfc4lf, precision_rfc4lf, recall_rfc4lf = obtain_metrics(test_labelapp_4l, y_pred_4l)
print('########### Métricas obtenidas con 6lf ###########' )
error_rate_rfc6lf, accuracy_rfc6lf, precision_rfc6lf, recall_rfc6lf = obtain_metrics(test_labelapp_6l, y_pred_6l)
print('########### Métricas obtenidas con 5f ###########' )
error_rate_rfc5f, accuracy_rfc5f, precision_rfc5f, recall_rfc5f = obtain_metrics(test_labelapp_5, y_pred_5)
print('########### Métricas obtenidas con 7f ###########' )
error_rate_rfc7f, accuracy_rfc7f, precision_rfc7f, recall_rfc7f = obtain_metrics(test_labelapp_7, y_pred_7)
print('########### Métricas obtenidas con 8f ###########' )
error_rate_rfc8f, accuracy_rfc8f, precision_rfc8f, recall_rfc8f = obtain_metrics(test_labelapp_8, y_pred_8)
print('########### Métricas obtenidas con 10f ###########' )
error_rate_rfc10f, accuracy_rfc10f, precision_rfc10f, recall_rfc10f = obtain_metrics(test_labelapp_10, y_pred_10)

# plot_confussion_matrix('rfc-8topfeatures', test_labelapp, y_pred)

# Plot performance vs num. features
figsize(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(['4lf','6lf','5f','7f','8f','10f','11f'], [error_rate_rfc4lf, error_rate_rfc6lf, error_rate_rfc5f, error_rate_rfc7f, error_rate_rfc8f, error_rate_rfc10f, error_rate_rfc11f], label='Rate Error')
plt.plot(['4lf','6lf','5f','7f','8f','10f','11f'], [accuracy_rfc4lf, accuracy_rfc6lf, accuracy_rfc5f, accuracy_rfc7f, accuracy_rfc8f, accuracy_rfc10f, accuracy_rfc11f], label='Accuracy')
plt.xlabel('Num. Features')
plt.ylabel('')
plt.legend()
plt.title('Performance vs Number of features')
plt.savefig('numfeaturesvsperformance.png')
plt.show()

print(
    '############################## ELABORACION DE GRILLA DE BUSQUEDA ALEATORIA CON CV ##############################')
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 12].values
'####################################### Separacion y estandarizacion de datos #######################################'

train_features, test_features, train_labelapp, test_labelapp = separe_standarize_data(X, y)

clf1_rfc = RandomForestClassifier(random_state=15)

# Visualizar parámetros usados por nuestro bosque
print('Parametros usados por el clasificador de Bosque Aleatorio:\n')
print(clf1_rfc.get_params('\n'))

# Numero de arboles en el bosque aleatorio
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Funcion para medir calidad de las divisiones
criterion = ['gini', 'entropy']
# Numero de features a considerar en cada division
max_features = ['auto', 'sqrt', 'log2']
# Numero maximo de niveles en el arbol
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Numero minimo de muestras requeridas para dividir un nodo
min_samples_split = [2, 5, 10]
# Numero minimo de muestras requeridas en cada nodo hoja
min_samples_leaf = [1, 2, 4]
# Metodo de seleccion de muestras para el entrenamiento de cada arbol
bootstrap = [True, False]

# Crear la grilla aleatoria
random_grid = {'n_estimators': n_estimators,
               'criterion': criterion,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint('\n')
# Mostrar grilla
print('Grilla de búsqueda aleatoria:')
pprint(random_grid)

pprint('\n')
pprint('##################################### Buscando mejores hiperparametros: #####################################')
# Usamos la grilla aleatoria para buscar los mejores hiperparámetros
# Primero, creamos el modelo base a tunear
clf_rfc = RandomForestClassifier(random_state=15)
# Luego, realizamos una búsqueda aleatoria de parámetros, utilizando la validación cruzada de 3 iteraciones
# Buscaremos a través de 100 combinaciones diferentes y usaremos todos los núcleos disponibles
# clf_rf_random = RandomizedSearchCV(estimator=clf_rfc, param_distributions=random_grid,
#                                   n_iter=1000, cv=4, verbose=2, random_state=15, n_jobs=-1)
clf_rf_random = RandomizedSearchCV(estimator=clf_rfc, param_distributions=random_grid,
                                   n_iter=200, cv=4, verbose=2, random_state=15, n_jobs=-1,
                                   return_train_score=True, scoring='neg_mean_absolute_error')
# clf_rf_random = RandomizedSearchCV(estimator=clf_rfc, param_distributions=random_grid, n_iter=500, cv=8, verbose=4,
# random_state=15, n_jobs=-1)
# Entrenaremos el modelo de búsqueda aleatoria
clf_rf_random.fit(train_features, train_labelapp)

# Mostrar todos los resultados de CV y ordenarlos según el rendimiento de prueba
random_results = pd.DataFrame(clf_rf_random.cv_results_).sort_values('mean_test_score', ascending=False)
# Mostrar los 30 primeros resultados
print(random_results.head(30))

# Ver mejores parámetros al ajustar la búsqueda aleatoria
pprint('Mejores hiperparametros:')
pprint(clf_rf_random.best_params_)
pprint('\n')

# Vamos a comparar un modelo base vs el mejor modelo de búsqueda aleatoria
# Modelo base: este modelo contiene 1000 arboles cada uno con un nivel de 100; usa el metodo de seleccion de bootstrap
# y el criterio gini
base_model_clf1 = RandomForestClassifier(n_estimators=1000, criterion='gini',
                                         max_features='auto', max_depth=100,
                                         min_samples_split=5, min_samples_leaf=2,
                                         bootstrap=True, random_state=15)
# Entrenar modelo base que usa 11f ahora con 3 nuevos parametros de max_features, min_samples_split, min_samples_leaf
base_model_clf1.fit(train_features, train_labelapp)
print('########################## MODELO DE CLASIFICACION 3: RANDOM FOREST BASE OPTIMIZADO ##########################')
print(base_model_clf1)
# Calculo de metricas: Exactitud, Precision, Sensibilidad, Valor F
y_prec_b, b_error, b_accuracy, b_precision, b_recall, b_f1 = evaluate('Modelo Base Optimizado RF', base_model_clf1,
                                                                      test_features,
                                                                      test_labelapp)
# Dibujar matriz de confusión de modelo base
plot_confussion_matrix('rfc-11features-optimized', test_labelapp, y_prec_b)

print('############################ MODELO DE CLASIFICACION 4: RANDOM GRID SEARCH CON CV #############################')
# Mejor modelo de búsqueda aleatoria que usara los mejores hiperparametros obtenidos
best_random = clf_rf_random.best_estimator_
pprint(best_random)
# Calculo de metricas del modelo. Este modelo ya se entreno previamente. Ver la linea 333.
y_pred_rd, rd_error, rd_accuracy, rd_precision, rd_recall, rd_f1 = evaluate('Mejor modelo de Random Search con CV',
                                                                            best_random, test_features, test_labelapp)
# Dibujar matriz de confusión de mejor modelo de Random Grid Search
plot_confussion_matrix('rfc-RandomGridSearchCV', test_labelapp, y_pred_rd)

# Mostrar mejoras en metricas
print('Improvement (Error Rate) of {:0.6f}%.'.format(100 * (rd_error - b_error) / b_error))
print('Improvement (Accuracy) of {:0.6f}%.'.format(100 * (rd_accuracy - b_accuracy) / b_accuracy))
print('Improvement (Precision) of {:0.6f}%.'.format(100 * (rd_precision - b_precision) / b_precision))
print('Improvement (Recall) of {:0.6f}%.'.format(100 * (rd_recall - b_recall) / b_recall))
print('Improvement (F1 Score) of {:0.6f}%.'.format(100 * (rd_f1 - b_f1) / b_f1))

print('################################## ELABORACION DE GRILLA DE BUSQUEDA CON CV ##################################')

# Crear grilla de parametros basado en los resultados de la búsqueda aleatoria
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': ['log2'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'criterion': ['gini'],
    'n_estimators': [400, 1000, 1400, 1800]
}
# Crear rango de arboles a evaluar
trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 1000, 1400]}

# Instanciar modelo de grid search
# grid_search = GridSearchCV(estimator=best_random, param_grid=trees_grid, cv=4, n_jobs=-1, verbose=2)
grid_search = GridSearchCV(estimator=best_random, param_grid=trees_grid, cv=6,
                           n_jobs=-1, verbose=2, scoring='neg_mean_absolute_error',
                           return_train_score=True)

# Ajustar la grilla de busqueda a los datos
grid_search.fit(train_features, train_labelapp)

# Mostrar resultados de Grid Search
grid_results = pd.DataFrame(grid_search.cv_results_)
print(grid_results.head(30))

print('################################ MODELO DE CLASIFICACION 5: GRID SEARCH CON CV ################################')

# Obtener mejores parametros
print('Mejores parametros de la grilla de busqueda', grid_search.best_params_)
# Obtener mejor estimador de la grilla de busqueda
best_grid = grid_search.best_estimator_
y_pred_gs, grid_err, grid_acc, grid_prec, grid_recall, grid_f1, = evaluate('Mejor modelo de Grid Search con CV',
                                                                           best_grid, test_features, test_labelapp)
# Dibujar matriz de confusión de mejor modelo de Grid Search
plot_confussion_matrix('rfc-GridSearchCV', test_labelapp, y_pred_gs)

# Mostrar mejoras en metricas
print('Improvement (Error Rate) of {:0.6f}%.'.format(100 * (grid_err - b_error) / b_error))
print('Improvement (Accuracy) of {:0.6f}%.'.format(100 * (grid_acc - b_accuracy) / b_accuracy))
print('Improvement (Precision) of {:0.6f}%.'.format(100 * (grid_prec - b_precision) / b_precision))
print('Improvement (Recall) of {:0.6f}%.'.format(100 * (grid_recall - b_recall) / b_recall))
print('Improvement (F1 Score) of {:0.6f}%.'.format(100 * (grid_f1 - b_f1) / b_f1))

print('################################ MODELO DE CLASIFICACION 6: SOPORTE VECTORIAL ################################')
from svc import *

error_rate_svc, accuracy_svc, precision_svc, recall_svc = svc_model()

'############################ Comparacion de resultados entre 5 modelos de Bosque Aleatorio ###########################'
# Los modelos son:
# (1)Modelo Base con 11 features de entrada y 5 parametros (n_estimators, criterion, max_depth, bootstrap, random_state)
# (2)Modelo con 8 features de entrada y los 5 anteriores parametros
# (3)Modelo Base con 11 features de entrada y 8 parametros (n_estimators, criterion, max_features, max_depth,
# min_samples_split, min_samples_leaf, bootstrap, random_state)
# (4)Mejor modelo obtenido con Random Grid Search con 11 features de entrada y mismos parametros de (3).
# (5)Mejor modelo obtenido con Grid Search con 11 features de entrada y mismos parametros de (3)
# (6)Modelo de Soporte Vectorial con 11 features de entrada y misma semilla que
# Las tecncias de los modelos (4) y (5) usaron Validacion Cruzada (CV - Cross Validation).

# Se compararan las metricas de Ratio de Error, Exactitud, Precision.
# (i) Comparacion del Ratio de error

plt.style.use('fivethirtyeight')
figsize(8, 8)
# Dataframe para agrupar resultados
error_comparison = pd.DataFrame({'model': ['8f RF Model', '11f RF Base Model', '11f RF Optimized Base Model',
                                           'Random Grid Search Model', 'Grid Search Model', 'SV Model'],
                                 'error_rate': [error_rate_rfc11f, error_rate_rfc8f, b_error,
                                                rd_error, grid_err, error_rate_svc]})
# Barra horizontal del grafico sobre evaluacion del ratio de error
error_comparison.sort_values('error_rate', ascending=False).plot(x='model', y='error_rate', kind='barh',
                                                                 color='blue', edgecolor='black')
# Ajustes de grafico
plt.ylabel('')
plt.yticks(size=12)
plt.xlabel('Error Rate')
plt.xticks(size=12)
plt.title('Model Comparison on Test Error Rate', size=14)
plt.savefig('plot-comparacion-ratioerror.png')
plt.show()

# (ii) Comparacion de la Exactitud
plt.style.use('fivethirtyeight')
figsize(8, 8)
# Dataframe para agrupar resultados
accuracy_comparison = pd.DataFrame({'model': ['8f RF Model', '11f RF Base Model', '11f RF Optimized Base Model',
                                              'Random Grid Search Model', 'Grid Search Model', 'SV Model'],
                                    'accuracy': [accuracy_rfc11f, accuracy_rfc8f, b_accuracy,
                                                 rd_accuracy, grid_acc, accuracy_svc]})

# Barra horizontal del grafico sobre evaluacion de la exactitud
accuracy_comparison.sort_values('accuracy', ascending=False).plot(x='model', y='accuracy', kind='barh',
                                                                  color='blue', edgecolor='black')
# Ajustes de grafico
plt.ylabel('')
plt.yticks(size=12)
plt.xlabel('Accuracy')
plt.xticks(size=12)
plt.title('Model Comparison on Test Accuracy', size=14)
plt.savefig('plot-comparacion-exactitud.png')
plt.show()

# (iii) Comparacion de la Sensibilidad
plt.style.use('fivethirtyeight')
figsize(8, 8)
# Dataframe para agrupar resultados
accuracy_comparison = pd.DataFrame({'model': ['8f RF Model', '11f RF Base Model', '11f RF Optimized Base Model',
                                              'Random Grid Search Model', 'Grid Search Model', 'SV Model'],
                                    'recall': [recall_rfc11f, recall_rfc8f, b_recall,
                                               rd_recall, grid_recall, recall_svc]})

# Barra horizontal del grafico sobre evaluacion de la sensibilidad
accuracy_comparison.sort_values('recall', ascending=False).plot(x='model', y='recall', kind='barh',
                                                                color='blue', edgecolor='black')
# Ajustes de grafico
plt.ylabel('')
plt.yticks(size=12)
plt.xlabel('Recall')
plt.xticks(size=12)
plt.title('Model Comparison on Test Recall', size=14)
plt.savefig('plot-comparacion-sensibilidad.png')
plt.show()

'########################## Grafico del error de entrenamiento y test vs el número de árboles #########################'
figsize(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(grid_results['param_n_estimators'], -1 * grid_results['mean_test_score'], label='Testing Error')
plt.plot(grid_results['param_n_estimators'], -1 * grid_results['mean_train_score'], label='Training Error')
plt.xlabel('Number of Trees')
plt.ylabel('Mean Abosolute Error')
plt.legend()
plt.title('Performance vs Number of Trees')
plt.savefig('plot-TrainingTestErrorVSnumarboles.png')
plt.show()


# Mostrar resultados de Grid Search ordenados por el score promedio de prueba
print(grid_results.sort_values('mean_test_score', ascending=False).head(15))

'################# Grafico Dist. valores reales y los valores pronosticados en el conjunto de pruebas #################'
figsize(8, 8)
# Grafico de densitdad de las predicciones y valores reales
sns.kdeplot(y_pred_gs, label='Predictions')
sns.kdeplot(test_labelapp, label='Actual Values')

# Asignando etiquetas a partes del grafico
plt.xlabel('Classification Score')
plt.ylabel('Density')
plt.title('Test Values and Predictions')
plt.savefig('plot-dist-actual-predicted-values.png')
plt.show()


'################################################### PCA Scatterplot ##################################################'
'''
#the following code makes a fancy looking plot using PC1 and PC2
pca_df = pd.DataFrame(train_features, index=[train_features.iloc[:, 1:3].values], columns=labels)
plt.scatter(pca_df.PC1, pca_df.PC2)
#plt.scatter(pca_df.PC1, pca_df.PC2, pca_df.PC3, pca_df.PC4)
#plt.scatter(pca_df.PC1, pca_df.PC2, pca_df.PC3, pca_df.PC4, pca_df.PC5, pca_df.PC6, pca_df.PC7 )
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(percentage_variation[0]))
plt.ylabel('PC2 - {0}%'.format(percentage_variation[1]))
# add samples
for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

plt.show()
'''
