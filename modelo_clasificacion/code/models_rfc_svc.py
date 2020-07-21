__title__ = ''
__author__ = 'Claudio Mori'
__credits__ = 'Sayo Makinwa'
__copyright__ = 'Copyright 2020, Thesis Project'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')
import seaborn as sns

sns.set(font_scale=3)
from IPython.core.pylabtools import figsize
from sklearn_evaluation import plot
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import precision_score as precision
from sklearn.metrics import recall_score as recall
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

# importar script creado para reutilizar funciones separe_standarize_data, obtain_metrics, plot_confussion_matrix
from preproc_metrics import *
# importar script para reutilizar función evaluate con el modelo base, de Random Grid Search y Grid search.
# funcion: evaluate(nombre_modelo, modelo_construido, features_prueba, labels_prueba)
from gs_metrics import *

pd.set_option('display.max_columns', None)

# Cargando dataset representado por archivo feature_vectors_counts.csv
feature_of_counts = "/home/lechu/Documents/GitHub/Prueba-de-concepto/seleccion_caracteristicas/processed_data" \
                    "/feature_vectors_counts.csv"

# Importando el dataset en forma de archivo .csv.
# Tiene 12 columnas: 11 para mostrar la frecuencia de aparición de los 11 feature extraídos de la aplicacióny 1 para
# la variable target está representada con el label o etiqueta que indica si la aplicación es malware o benigna.
dataset = pd.read_csv(feature_of_counts, index_col=0)

# X: 11 tipos de características (features)
# permisos solicitaciones, llamadas api restringidas, actividades
# llamadas api sospechosas, permisos usados, servicios
# filtros intent, receptor de mensajes, uses-feature
# urls, proveedor de contenido
X = dataset.iloc[:, 1:12].values
print(X.shape)
print(X, '\n')

# y: etiqueta de la aplicación (0: benigna, 1: malware) (variable target)
y = dataset.iloc[:, 12].values
print(y.shape)
print(y, '\n')

# Separacion de datos y Estandarizacion de los mismos
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
'###################################### MODELO DE CLASIFICACION: RANDOM FOREST ####################################'
# Random Forest usando 11 features
clf_rfc_11f = RandomForestClassifier(random_state=15)
# Entrenamiento de modelo
clf_rfc_11f.fit(train_features, train_labelapp)
# Obtención de predicciones
y_train_predicted_rfc11f = clf_rfc_11f.predict(train_features)
y_test_predicted_rfc11f = clf_rfc_11f.predict(test_features)

'################################### MODELO DE CLASIFICACION: SOPORTE VECTORIAL ###################################'
# Soporte Vectorial usando 11 features
clf_svc_11f = SVC(random_state=15)
# Entrenamiento del modelo
clf_svc_11f.fit(train_features, train_labelapp)
# Obtención de predicciones
y_train_pred_svc11f = clf_svc_11f.predict(train_features)
y_test_pred_svc11f = clf_svc_11f.predict(test_features)

print('################################# Resultados de los modelos usando 11 features ################################')
print('######################################## Métricas de Random Forest ########################################')
error_rate_rfc11f, accuracy_rfc11f, precision_rfc11f, recall_rfc11f, f1_rfc11f = obtain_metrics(test_labelapp,
                                                                                     y_test_predicted_rfc11f)
plot_confussion_matrix('rfc-11features', test_labelapp, y_test_predicted_rfc11f)
print('###################################### Métricas de Soporte Vectorial ######################################')
error_rate_svc11f, accuracy_svc11f, precision_svc11f, recall_svc11f, f1_svc11f = obtain_metrics(test_labelapp, y_test_pred_svc11f)
plot_confussion_matrix('svc_11features', test_labelapp, y_test_pred_svc11f)

print('########################## MODELO DE CLASIFICACION USANDO FEATURES SELECCIONADOS ###########################')
print('############################# Encontrando features importantes con Scikit-learn ############################')
clf = RandomForestClassifier(random_state=15)
clf.fit(train_features, train_labelapp)

feature_names = list(dataset.iloc[:, 1:12].columns)
# print(feature_names)

# Calcular la importancia de cada uno de los 11 features
feature_importance = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)
print(feature_importance)

print(
    '################################## Grafico de barras con importantacia features ##################################')
figsize(30, 32)
# Crear barras de features del grafico
sns.barplot(x=feature_importance, y=feature_importance.index, seed=15)
# Añadir etiquetas de los features al grafico
plt.xlabel('Feature Importance Score')
plt.ylabel('11 Apks Features')
plt.title("Visualizing Important Features")
# plt.legend
plt.savefig('../code/plots/barras-importancia-11features.png')
plt.show()

'##################################### Random Forest con 11 features y SVC con 5 features  ######################################'
# Conjunto de 11 features para usar con RF
X = dataset.iloc[:, 1:12].values
# Conjunto de 5 primeros features (X_5) y de etiquetas (y)
X_5f = dataset[['permisos_solicitados', 'api_restringidas', 'actividades', 'api_sospechosas', 'permisos_usados']]
y = dataset.iloc[:, 12].values

# orden de features según puntuación
# X_f = dataset[['permisos_solicitados', 'api_restringidas', 'actividades', 'api_sospechosas', 'permisos_usados',
#              'servicios', 'filtros_intent', 'receptor_mensajes', 'urls', 'uses-feature', 'proveedor_contenido']]

# Separacion de datos y Estandarizacion de los mismos
# conjunto de datos de entrenamiento y prueba para usar con ...
# Random Forest
train_features_11, test_features_11, train_labelapp_11, test_labelapp_11 = separe_standarize_data(X, y)
# Soporte Vectorial
train_features_5, test_features_5, train_labelapp_5, test_labelapp_5 = separe_standarize_data(X_5f, y)

# Modelo de clasificacion: Random Forest usando 5 top features
# clf_rfc_11f = RandomForestClassifier(n_estimators=1000, criterion='gini', bootstrap=True, max_depth=100, random_state=15)
clf_rfc_11f = RandomForestClassifier(random_state=15)
# clf_svc_top5f = SVC(C=1.0, kernel='rbf', gamma=0.73, max_iter=900, random_state=15)
clf_svc_top5f = SVC(random_state=15)

# Entrenamiento de modelos
clf_rfc_11f.fit(train_features_11, train_labelapp_11)
clf_svc_top5f.fit(train_features_5, train_labelapp_5)

# Obtención de perdicciones
y_pred_rfc11f = clf_rfc_11f.predict(test_features_11)
y_pred_svc5f = clf_svc_top5f.predict(test_features_5)

print('########### Métricas de RF usando 11 features ###########')
error_rate_rfc11f, accuracy_rfc11f, precision_rfc11f, recall_rfc11f, f1_rfc11f = obtain_metrics(test_labelapp_11, y_pred_rfc11f)
plot_confussion_matrix('rfc-11features', test_labelapp_11, y_pred_rfc11f)

print('########### Métricas de SV usando 5 features más importantes ###########')
error_rate_svc5f, accuracy_svc5f, precision_svc5f, recall_svc5f, f1_svc5f = obtain_metrics(test_labelapp_5, y_pred_svc5f)
plot_confussion_matrix('svc_5f', test_labelapp_5, y_pred_svc5f)

'################################ Random Forest con 11 features y SVC con 6 features  #################################'
# Conjunto de 11 features para usar con RF
X = dataset.iloc[:, 1:12].values
# Conjunto de 6 primeros features (X_6) y de etiquetas (y)
X_6f = dataset[['permisos_solicitados', 'api_restringidas', 'actividades', 'api_sospechosas', 'permisos_usados',
                'servicios']]
y = dataset.iloc[:, 12].values

# Separacion de datos y Estandarizacion de los mismos
# conjunto de datos de entrenamiento y prueba para usar con ...
# Random Forest
train_features_11, test_features_11, train_labelapp_11, test_labelapp_11 = separe_standarize_data(X, y)
# Soporte Vectorial
train_features_6, test_features_6, train_labelapp_6, test_labelapp_6 = separe_standarize_data(X_6f, y)

# Modelo de clasificacion: Random Forest usando 5 top features
# clf_rfc_11f = RandomForestClassifier(n_estimators=1000, criterion='gini', bootstrap=True, max_depth=100, random_state=15)
clf_rfc_11f = RandomForestClassifier(random_state=15)
# clf_svc_top6f = SVC(C=1.0, kernel='rbf', gamma=0.73, max_iter=900, random_state=15)
clf_svc_top6f = SVC(random_state=15)

# Entrenamiento de modelos
clf_rfc_11f.fit(train_features_11, train_labelapp_11)
clf_svc_top6f.fit(train_features_6, train_labelapp_6)

# Obtención de perdicciones
y_pred_rfc11f = clf_rfc_11f.predict(test_features_11)
y_pred_svc6f = clf_svc_top6f.predict(test_features_6)

print('########### Métricas de RF usando 11 features ###########')
error_rate_rfc11f, accuracy_rfc11f, precision_rfc11f, recall_rfc11f, f1_rfc11f = obtain_metrics(test_labelapp_11, y_pred_rfc11f)
plot_confussion_matrix('rfc-11features', test_labelapp_11, y_pred_rfc11f)

print('########### Métricas de SV usando 6 features más importantes ###########')
error_rate_svc6f, accuracy_svc6f, precision_svc6f, recall_svc6f, f1_svc6f = obtain_metrics(test_labelapp_6, y_pred_svc6f)
plot_confussion_matrix('svc_6f', test_labelapp_6, y_pred_svc6f)

'######################### Modelos de Random Forest usando diferentes cantidades de features  #########################'
# Crear modelo en base a las características más relevantes

# ELEGIR UNO DE LOS CONJUNTOS DE FEATURES PARA USARLO EN EL ENTRENAMIENTO DEL MODELO
# low 4 features
X_4l = dataset[['proveedor_contenido', 'urls', 'uses-feature', 'receptor_mensajes']]
# low 6 features
X_6l = dataset[['proveedor_contenido', 'urls', 'uses-feature', 'receptor_mensajes', 'filtros_intent', 'servicios']]
# top 5 features
X_5f = dataset[['permisos_solicitados', 'api_restringidas', 'actividades', 'api_sospechosas', 'permisos_usados']]
# top 6 features
X_6 = dataset[['permisos_solicitados', 'api_restringidas', 'actividades', 'api_sospechosas', 'permisos_usados',
               'servicios']]
# top 8 features
X_8 = dataset[['permisos_solicitados', 'api_restringidas', 'actividades', 'api_sospechosas', 'permisos_usados',
               'servicios', 'filtros_intent', 'receptor_mensajes']]
# top 10 features
X_10 = dataset[['permisos_solicitados', 'api_restringidas', 'actividades', 'api_sospechosas', 'permisos_usados',
                'servicios', 'filtros_intent', 'receptor_mensajes', 'urls', 'uses-feature']]
y = dataset.iloc[:, 12].values

# Separacion de datos y Estandarizacion de los mismos
train_features_4l, test_features_4l, train_labelapp_4l, test_labelapp_4l = separe_standarize_data(X_4l, y)
train_features_6l, test_features_6l, train_labelapp_6l, test_labelapp_6l = separe_standarize_data(X_6l, y)
train_features_5, test_features_5, train_labelapp_5, test_labelapp_5 = separe_standarize_data(X_5f, y)
train_features_6, test_features_6, train_labelapp_6, test_labelapp_6 = separe_standarize_data(X_6, y)
train_features_8, test_features_8, train_labelapp_8, test_labelapp_8 = separe_standarize_data(X_8, y)
train_features_10, test_features_10, train_labelapp_10, test_labelapp_10 = separe_standarize_data(X_10, y)

# Modelo de clasificacion 2: Random Forest usando 10 top features, 1000 arboles y con profundidad de 100 cada uno
clf_rfc_f = RandomForestClassifier(n_estimators=1000, criterion='gini', bootstrap=True, max_depth=100, random_state=15)
clf_rfc_low4f = RandomForestClassifier(random_state=15)
clf_rfc_low6f = RandomForestClassifier(random_state=15)
clf_rfc_top5f = RandomForestClassifier(random_state=15)
clf_rfc_top6f = RandomForestClassifier(random_state=15)
clf_rfc_top8f = RandomForestClassifier(random_state=15)
clf_rfc_top10f = RandomForestClassifier(random_state=15)

# Entrenamiento de modeloS
clf_rfc_low4f.fit(train_features_4l, train_labelapp_4l)
clf_rfc_low6f.fit(train_features_6l, train_labelapp_6l)
clf_rfc_top5f.fit(train_features_5, train_labelapp_5)
clf_rfc_top6f.fit(train_features_6, train_labelapp_6)
clf_rfc_top8f.fit(train_features_8, train_labelapp_10)
clf_rfc_top10f.fit(train_features_10, train_labelapp_10)

# Obtención de perdicciones
y_pred_4lf = clf_rfc_low4f.predict(test_features_4l)
y_pred_6lf = clf_rfc_low6f.predict(test_features_6l)
y_pred_5f = clf_rfc_top5f.predict(test_features_5)
y_pred_6f = clf_rfc_top6f.predict(test_features_6)
y_pred_8f = clf_rfc_top8f.predict(test_features_8)
y_pred_10f = clf_rfc_top10f.predict(test_features_10)

print('######################## Resultados del modelo usando features con mayor importancia #########################')
print('########### Métricas y matriz de confusion obtenidas para ###########')
print('########### 4 últimos features ###########')
error_rate_rfc4lf, accuracy_rfc4lf, precision_rfc4lf, recall_rfc4lf, f1_rfc4lf = obtain_metrics(test_labelapp_4l, y_pred_4lf)
plot_confussion_matrix('rfc_4lf', test_labelapp_4l, y_pred_4lf)
print('########### 6 últimos features ###########')
error_rate_rfc6lf, accuracy_rfc6lf, precision_rfc6lf, recall_rfc6lf, f1_rfc6lf = obtain_metrics(test_labelapp_6l, y_pred_6lf)
plot_confussion_matrix('rfc_6lf', test_labelapp_6l, y_pred_6lf)
print('########### 5 primeros features ###########')
error_rate_rfc5f, accuracy_rfc5f, precision_rfc5f, recall_rfc5f, f1_rfc5f = obtain_metrics(test_labelapp_5, y_pred_5f)
plot_confussion_matrix('rfc_5f', test_labelapp_5, y_pred_5f)
print('########### 6 primeros features ###########')
error_rate_rfc6f, accuracy_rfc6f, precision_rfc6f, recall_rfc6f, f1_rfc6f = obtain_metrics(test_labelapp_6, y_pred_6f)
plot_confussion_matrix('rfc_6f', test_labelapp_6, y_pred_6f)
print('########### 8 primeros features ###########')
error_rate_rfc8f, accuracy_rfc8f, precision_rfc8f, recall_rfc8f, f1_rfc8f = obtain_metrics(test_labelapp_8, y_pred_8f)
plot_confussion_matrix('rfc_8f', test_labelapp_8, y_pred_8f)
print('########### 10 primeros features ###########')
error_rate_rfc10f, accuracy_rfc10f, precision_rfc10f, recall_rfc10f, f1_rfc10f = obtain_metrics(test_labelapp_10, y_pred_10f)
plot_confussion_matrix('rfc_10f', test_labelapp_10, y_pred_10f)

'######################################## Gráfico Performance vs Num. Features ########################################'
figsize(20, 22)
plt.style.use('fivethirtyeight')
plt.plot(['4lf', '6lf', '5f', '6f', '8f', '10f', '11f'],
         [error_rate_rfc4lf, error_rate_rfc6lf, error_rate_rfc5f, error_rate_rfc6f, error_rate_rfc8f, error_rate_rfc10f,
          error_rate_rfc11f], label='Rate Error')
plt.plot(['4lf', '6lf', '5f', '6f', '8f', '10f', '11f'],
         [accuracy_rfc4lf, accuracy_rfc6lf, accuracy_rfc5f, accuracy_rfc6f, accuracy_rfc8f, accuracy_rfc10f,
          accuracy_rfc11f], label='Accuracy')
plt.xlabel('Num. Features')
plt.ylabel('')
plt.legend()
plt.title('Performance vs Number of features')
plt.savefig('../code/plots/plot-Performance-vs-numfeatures.png')
plt.show()

'############################# Gráfico Accuracy, Precision, Recall, F1 vs Num. Features ###############################'
figsize(20, 22)
plt.style.use('fivethirtyeight')
plt.plot(['4lf', '6lf', '5f', '6f', '8f', '10f', '11f'],
         [accuracy_rfc4lf, accuracy_rfc6lf, accuracy_rfc5f, accuracy_rfc6f, accuracy_rfc8f, accuracy_rfc10f,
          accuracy_rfc11f], label='Accuracy')
plt.plot(['4lf', '6lf', '5f', '6f', '8f', '10f', '11f'],
         [precision_rfc4lf, precision_rfc6lf, precision_rfc5f, precision_rfc6f, precision_rfc8f, precision_rfc10f,
          precision_rfc11f], label='Precision')
plt.plot(['4lf', '6lf', '5f', '6f', '8f', '10f', '11f'],
         [recall_rfc4lf, recall_rfc6lf, recall_rfc5f, recall_rfc6f, recall_rfc8f, recall_rfc10f,
          recall_rfc11f], label='Recall')
plt.plot(['4lf', '6lf', '5f', '6f', '8f', '10f', '11f'],
         [f1_rfc4lf , f1_rfc6lf,  f1_rfc5f, f1_rfc6f, f1_rfc8f, f1_rfc10f,
          f1_rfc11f], label='F1-Score')
plt.xlabel('Num. Features')
plt.ylabel('')
plt.legend()
plt.title('Accuracy, Precision, Recall, F1-Score vs Number of features')
plt.savefig('../code/plots/plot-acc_prec_recall_f1-vs-numfeatures.png')
plt.show()

print('######################### TUNEO DE HIPERPARÁMETRO DE NUM. ÁRBOLES CON ITERACIONES #########################')
clf_rfc_11f = RandomForestClassifier(n_estimators=1, criterion='gini', max_depth=110, max_features='sqrt',
                                     min_samples_leaf=1, min_samples_split=19, bootstrap=True, random_state=15)
trees, train_loss, test_loss = [], [], []
# Afinación de hiperparámetro del número de árboles (n_estimators)
for iter in range(1089):
    # Entrenamiento del modelo
    clf_rfc_11f.fit(train_features, train_labelapp)
    # Obtención de predicciones
    y_train_pred_rfc = clf_rfc_11f.predict(train_features)
    y_test_pred_rfc = clf_rfc_11f.predict(test_features)
    # Cálculo de Error Cuadrado Medio (MSE)
    mse_train = mean_squared_error(train_labelapp, y_train_pred_rfc)
    mse_test = mean_squared_error(test_labelapp, y_test_pred_rfc)
    print("Iteration: {} Train mse: {} Test mse: {}".format(iter, mse_train, mse_test))
    # Acumulación de árboles
    trees += [clf_rfc_11f.n_estimators]
    train_loss += [mse_train]
    test_loss += [mse_test]
    # Aumento de árboles en uno
    clf_rfc_11f.n_estimators += 1
# Dibujar gráfica de número de árboles vs MSEs calculados.
# Se podrá ver si el overfitting incrementa o no cuando se añade más árboles
plt.figure(figsize=(22, 25))
plt.plot(trees, train_loss, color="blue", label="MSE on Train data")
plt.plot(trees, test_loss, color="red", label="MSE on Test data")
plt.ylabel("Mean Squared Error", fontsize='x-large');
plt.yticks(size=24, fontsize='x-large')
plt.xlabel("# of trees", fontsize='x-large')
plt.xticks(size=24, fontsize='x-large')
plt.savefig('../code/plots/plot-traintesterror-vs-numtrees.png')
plt.legend()
plt.show()

print(
    '############################## ELABORACION DE GRILLA DE BUSQUEDA ALEATORIA CON CV ##############################')
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 12].values

# Separacion y estandarizacion de datos
train_features, test_features, train_labelapp, test_labelapp = separe_standarize_data(X, y)

clf1_rfc = RandomForestClassifier(random_state=15)

from pprint import pprint

# Visualizar parámetros usados por nuestro bosque
print('Parametros usados por el clasificador de Bosque Aleatorio:\n')
print(clf1_rfc.get_params('\n'))

# Numero de arboles en el bosque aleatorio
n_estimators = [int(x) for x in np.linspace(start=1, stop=1400, num=10)]
# Funcion para medir calidad de las divisiones
criterion = ['gini', 'entropy']
# Numero de features a considerar en cada division
max_features = ['sqrt', 'log2']
# Numero maximo de niveles en el arbol
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Numero minimo de muestras requeridas para dividir un nodo
min_samples_split = [2, 5, 10, 20]
# Numero minimo de muestras requeridas en cada nodo hoja
min_samples_leaf = [1, 2, 4, 7, 10]
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
# Buscaremos a través de combinaciones diferentes y usaremos todos los núcleos disponibles
clf_rf_random = RandomizedSearchCV(estimator=clf_rfc, param_distributions=random_grid,
                                   n_iter=200, cv=4, verbose=5, random_state=15, n_jobs=4,
                                   scoring='neg_mean_absolute_error', return_train_score=True, )
# 500 iteaciones y 8 subconjuntos
# clf_rf_random = RandomizedSearchCV(estimator=clf_rfc, param_distributions=random_grid, n_iter=500, cv=8, verbose=4,
# random_state=15, n_jobs=-1)

# Entrenaremos el modelo de búsqueda aleatoria
clf_rf_random.fit(train_features, train_labelapp)

# Mostrar todos los resultados de CV y ordenarlos según el rendimiento de prueba
random_results = pd.DataFrame(clf_rf_random.cv_results_)
# Mostrar los 30 primeros resultados
print(random_results.head(30))

# Ver mejores parámetros al ajustar la búsqueda aleatoria
pprint('Mejores hiperparametros obtenidos con Random Grid Search:')
pprint(clf_rf_random.best_params_)
pprint('Mejor estimador obtenido con Random Grid Search:')
pprint(clf_rf_random.best_estimator_)
pprint('\n')

# Vamos a comparar un modelo base vs el mejor modelo de búsqueda aleatoria
# Modelo base: este modelo contiene 1000 arboles cada uno con un nivel de 100; usa el metodo de seleccion de bootstrap
# y el criterio gini
# base_model_clf1 = RandomForestClassifier(n_estimators=1000, criterion='gini',
#                                         max_features='auto', max_depth=100,
#                                         min_samples_split=5, min_samples_leaf=2,
#                                         bootstrap=True, random_state=15)
base_model_clf1 = RandomForestClassifier(random_state=15)

# Entrenar modelo base que usa 11f ahora con 3 nuevos parametros de max_features, min_samples_split, min_samples_leaf
base_model_clf1.fit(train_features, train_labelapp)

print('########################## MODELO DE CLASIFICACION: RANDOM FOREST BASE OPTIMIZADO ##########################')
print(base_model_clf1)
# Calculo de metricas: Exactitud, Precision, Sensibilidad, Valor F
y_pred_b, b_error, b_accuracy, b_precision, b_recall, b_f1 = evaluate(' Modelo Base RF', base_model_clf1,
                                                                      test_features,
                                                                      test_labelapp)
plot_confussion_matrix('rfc_11f', test_labelapp, y_pred_b)

print('############################ MODELO DE CLASIFICACION: RANDOM GRID SEARCH CON CV #############################')
# Mejor modelo de búsqueda aleatoria que usara los mejores hiperparametros obtenidos
best_random = clf_rf_random.best_estimator_
pprint(best_random)
# Calculo de metricas del modelo. Este modelo ya se entreno previamente.
y_pred_rd, rd_error, rd_accuracy, rd_precision, rd_recall, rd_f1 = evaluate(' Mejor modelo de Random Search con CV',
                                                                            best_random, test_features, test_labelapp)
plot_confussion_matrix('random-grid-search_rfc11f', test_labelapp, y_pred_rd)

print('\n')
print('Improvement (Error Rate) of {:0.2f}%.'.format(100 * (rd_error - b_error) / b_error))
print('Improvement (Accuracy) of {:0.2f}%.'.format(100 * (rd_accuracy - b_accuracy) / b_accuracy))
print('Improvement (Precision) of {:0.2f}%.'.format(100 * (rd_precision - b_precision) / b_precision))
print('Improvement (Recall) of {:0.2f}%.'.format(100 * (rd_recall - b_recall) / b_recall))
print('Improvement (F1 Score) of {:0.2f}%.'.format(100 * (rd_f1 - b_f1) / b_f1))

print('################################## ELABORACION DE GRILLA DE BUSQUEDA CON CV ##################################')

# Crear grilla de parametros basado en los resultados de la búsqueda aleatoria
param_grid = {
    'bootstrap': [True],
    'max_depth': [40, 60, 80, 90],
    'max_features': ['log2'],
    'min_samples_leaf': [2, 3, 4],
    'min_samples_split': [4, 6, 8],
    'criterion': ['gini'],
    'n_estimators': [100, 200, 400, 600, 800, 1000]
}

# Crear rango de arboles a evaluar
# trees_grid = {'n_estimators': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 1000, 1400]}
# Modelo base de Bosque Aleatorio
clf_rfc = RandomForestClassifier(random_state=15)
# Instanciar modelo de grid search
grid_search = GridSearchCV(estimator=clf_rfc, param_grid=param_grid, cv=6,
                           n_jobs=4, verbose=10, scoring='neg_mean_absolute_error',
                           return_train_score=True)

# Ajustar la grilla de busqueda a los datos
grid_search.fit(train_features, train_labelapp)

# Mostrar resultados de Grid Search
grid_results = pd.DataFrame(grid_search.cv_results_)
print(grid_results)
# print(grid_results.sort_values('mean_test_score', ascending=False).head(15))

print('################################ MODELO DE CLASIFICACION: GRID SEARCH CON CV ################################')

# Obtener mejor estimador de la grilla de busqueda
best_grid = grid_search.best_estimator_
print('Mejores hiperparámetros obtenidos con Grid Search:')
print(grid_search.best_params_)
print('Mejor estimador obtenido con Grid Search:')
print(grid_search.best_estimator_)

y_pred_gs, grid_err, grid_acc, grid_prec, grid_recall, grid_f1, = evaluate('Mejor modelo de Grid Search con CV',
                                                                           best_grid, test_features, test_labelapp)
# Dibujar matriz de confusión de mejor modelo de Grid Search
plot_confussion_matrix('grid-search_rfc11f', test_labelapp, y_pred_gs)

# Mostrar mejoras en metricas
print('\n')
print('Improvement (Error Rate) of {:0.2f}%.'.format(100 * (grid_err - b_error) / b_error))
print('Improvement (Accuracy) of {:0.2f}%.'.format(100 * (grid_acc - b_accuracy) / b_accuracy))
print('Improvement (Precision) of {:0.2f}%.'.format(100 * (grid_prec - b_precision) / b_precision))
print('Improvement (Recall) of {:0.2f}%.'.format(100 * (grid_recall - b_recall) / b_recall))
print('Improvement (F1 Score) of {:0.2f}%.'.format(100 * (grid_f1 - b_f1) / b_f1))

'############################ Comparacion de resultados entre 5 modelos de Bosque Aleatorio ###########################'
# Los modelos son:
# (1)Modelo RF usando 5 features con mejor puntuación y como único parámetro a random_state
# (2)Modelo RF usando 6 features con mejor puntuación y como único parámetro a random_state
# (3)Modelo RF usando los 11 features y como único parámetro a random_state
# (4)Modelo SVM usando 5 features con mejor puntuación y único parámetro random_statte
# (5)Modelo SVM usando 6 features con mejor puntuación y como único parámetro a random_state
# (6)Modelo SVM usando los 11 features y como único parámetro a random_state
# (7)Modelo RF obtenido con Random Grid Search con CV usando los 11 features y los parametros tuneados de
# (n_estimators, criterion, max_features, max_depth, min_samples_split, min_samples_leaf, bootstrap, random_state)
# (8)Modelo RF obtenido con Grid Search con CV usando los 11 features y habiendo tuneado los anteriores 7 parámetros.
# Las tecncias de los modelos (6) y (7) usaron Validacion Cruzada (CV - Cross Validation).

# Se compararan las metricas de Ratio de Error, Exactitud, Precision.
# (i) Comparacion del Ratio de error

plt.style.use('fivethirtyeight')
figsize(12, 14)
# Dataframe para agrupar resultados
error_comparison = pd.DataFrame({'model': ['5f RF Model', '6f RF Model', '11f RF Model',
                                           '5f SVC Model', '6f SVC Model', '11f SVC Model',
                                           'Random Grid Search Model', 'Grid Search Model'],
                                 'error_rate': [error_rate_rfc5f, error_rate_rfc6f, error_rate_rfc11f,
                                                error_rate_svc5f, error_rate_svc6f, error_rate_svc11f,
                                                rd_error, grid_err]})
# Barra horizontal del grafico sobre evaluacion del ratio de error
error_comparison.sort_values('error_rate', ascending=False).plot(x='model', y='error_rate', kind='barh',
                                                                 color='blue', edgecolor='black')
# Ajustes de grafico
plt.ylabel('')
plt.yticks(size=12, fontsize='x-large')
plt.xlabel('Error Rate', fontsize='xx-large')
plt.xticks(size=12, fontsize='x-large')
plt.title('Model Comparison on Test Error Rate', size=24)
plt.savefig('../code/plots/plot-comparacion-ratioerror.png')
plt.show()

# (ii) Comparacion de la Exactitud
plt.style.use('fivethirtyeight')
figsize(12, 14)
# Dataframe para agrupar resultados
accuracy_comparison = pd.DataFrame({'model': ['5f RF Model', '6f RF Model', '11f RF Model',
                                              '5f SVC Model', '6f SVC Model', '11f SVC Model',
                                              'Random Grid Search Model', 'Grid Search Model'],
                                    'accuracy': [accuracy_rfc5f, accuracy_rfc6f, accuracy_rfc11f,
                                                 accuracy_svc5f, accuracy_svc6f, accuracy_svc11f,
                                                 rd_accuracy, grid_acc]})

# Barra horizontal del grafico sobre evaluacion de la exactitud
accuracy_comparison.sort_values('accuracy', ascending=False).plot(x='model', y='accuracy', kind='barh',
                                                                  color='blue', edgecolor='black')
# Ajustes de grafico
plt.ylabel('')
plt.yticks(size=12, fontsize='x-large')
plt.xlabel('Accuracy', fontsize='xx-large')
plt.xticks(size=12, fontsize='x-large')
plt.title('Model Comparison on Test Accuracy', size=24)
plt.savefig('../code/plots/plot-comparacion-exactitud.png')
plt.show()

# (iii) Comparacion de la Sensibilidad
plt.style.use('fivethirtyeight')
figsize(12, 14)
# Dataframe para agrupar resultados
accuracy_comparison = pd.DataFrame({'model': ['5f RF Model', '6f RF Model', '11f RF Model',
                                              '5f SVC Model', '6f SVC Model', '11f SVC Model',
                                              'Random Grid Search Model', 'Grid Search Model'],
                                    'recall': [recall_rfc5f, recall_rfc6f, recall_rfc11f,
                                               recall_svc5f, recall_svc6f, recall_svc11f,
                                               rd_recall, grid_recall]})

# Barra horizontal del grafico sobre evaluacion de la sensibilidad
accuracy_comparison.sort_values('recall', ascending=False).plot(x='model', y='recall', kind='barh',
                                                                color='blue', edgecolor='black')
# Ajustes de grafico
plt.ylabel('')
plt.yticks(size=12, fontsize='x-large')
plt.xlabel('Recall', fontsize='xx-large')
plt.xticks(size=12, fontsize='x-large')
plt.title('Model Comparison on Test Recall', size=24)
plt.savefig('../code/plots/plot-comparacion-sensibilidad.png')
plt.show()

'########################## Grafico del error de entrenamiento y test vs el número de árboles #########################'
plot_results(clf_rf_random)
plot_results(grid_search)

'''
figsize(8, 8)
plt.style.use('fivethirtyeight')
plt.plot(grid_results['param_n_estimators'], -1 * grid_results['mean_test_score'], label='Testing Error')
plt.plot(grid_results['param_n_estimators'], -1 * grid_results['mean_train_score'], label='Training Error')
plt.xlabel('Number of Trees')
plt.ylabel('Mean Abosolute Error')
plt.legend()
plt.title('Performance vs Number of Trees')
plt.savefig('plot-TrainingTestErrorVSNumArboles.png')
plt.show()
'''
'################# Grafico Dist. valores reales y los valores pronosticados en el conjunto de pruebas #################'
figsize(20, 22)
# Grafico de densitdad de las predicciones y valores reales
sns.kdeplot(y_pred_gs, label='Predictions')
sns.kdeplot(test_labelapp, label='Actual Values')

# Asignando etiquetas a partes del grafico
plt.xlabel('Classification Score', fontsize='xx-large')
plt.xticks(size=12, fontsize='x-large')
plt.ylabel('Density', fontsize='xx-large')
plt.yticks(size=12, fontsize='x-large')
plt.title('Test Values and Predictions', size=24)
plt.savefig('../code/plots/plot-dist-actual-predicted-values.png')
plt.show()

'########################################### Curva ROC de modelos generados ###########################################'
fig, ax = plt.subplots()
plot.roc(test_labelapp, y_pred_svc5f, ax=ax)
plot.roc(test_labelapp, y_pred_5f, ax=ax)
plot.roc(test_labelapp, y_test_pred_svc11f, ax=ax)
plot.roc(test_labelapp, y_test_predicted_rfc11f, ax=ax)
plot.roc(test_labelapp, y_pred_gs, ax=ax)
plot.roc(test_labelapp, y_pred_rd, ax=ax)

ax.legend(['SVM 5f Model', 'RF 5f Model', 'SVM 11f Model', 'RF 11f Model', 'RF Grid Search Model',
           'RF Random Grid Search Model'], fontsize='small')
fig.savefig('../code/plots/Curva-ROC.png')
plt.show()
fig

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
