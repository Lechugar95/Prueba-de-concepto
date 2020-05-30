# coding=utf-8
# import libraries
import pandas as pd
import numpy as np
import os
import csv
import random

# Set seed for random numbers
random.seed(1)

# paths to files
# ruta del archivo feature_vectors_counts_temp.csv que contiene la cantidad de veces que aparece cada característica
# en el vector del apk
feature_of_counts_temp = "../processed_data/feature_vectors_counts_temp.csv"
# ruta del archivo de feature_vectors_counts.csv que contiene la información de feature_vectors_counts_temp.csv más
# la etiqueta de si es malware o no
feature_of_counts = "../processed_data/feature_vectors_counts.csv"
# feature_of_sets = "../processed_data/feature_vectors_sets.csv"
# directorio donde se encuentra la carpeta feature vectors con los archivos de características extraídas para cada apk.
# El dataset incluye archivos donde cada línea del archivo es una características extraída.
dir_of_files = "../raw_data/feature_vectors/"
# dir_of_files = "/home/lechu/Documents/UL/2020-1/PoC/datasets/Drebin/drebin/feature_vectors/"
# El dataset también incluye un archivo .csv con las aplicaciones que fueron detectadas como malware en base
# a escaneres de virustotal
# pertenecen
# known_malware_files = "../raw_data/sha256_family.csv"
known_malware_files = "../raw_data/malware_file.csv"

apk_total_features_id = {x: 0 for x in range(1, 12)}
# Usar con dataset de Drebin
'''apk_total_feature_name = \
    {
        "feature": 0,
        "permission": 0,
        "activity": 0,
        "service_receiver": 0,
        "provider": 0,
        "service": 0,
        "intent": 0,
        "api_call": 0,
        "real_permission": 0,
        "call": 0,
        "url": 0
    }'''
# Usar con cualquier otro dataset excepto el de Drebin
apk_total_feature_name = \
    {
        "componente_actividades": 0,
        "componente_proveedor_contenido": 0,
        "componente_receptor_mensajes": 0,
        "componente_servicios": 0,
        "dominios_url": 0,
        "filtros_intent": 0,
        "llamadas_api_restringidas": 0,
        "llamadas_api_sospechosas": 0,
        "permisos_solicitado": 0,
        "permisos_usados": 0,
        "uses_feature": 0
    }


# functions used


def count_feature_set(feature_file):
    # Usar con el dataset de Drebin
    """features_types_array = {
        "feature": 1,
        "permission": 2,
        "activity": 3,
        "service_receiver": 3,
        "provider": 3,
        "service": 3,
        "intent": 4,
        "api_call": 5,
        "real_permission": 6,
        "call": 7,
        "url": 8
    }"""

    # Usar con cualquier dataset excepto el de Drebin
    features_types_array = {
        "componente_actividades": 1,
        "componente_proveedor_contenido": 2,
        "componente_receptor_mensajes": 3,
        "componente_servicios": 4,
        "dominios_url": 5,
        "filtros_intent": 6,
        "llamadas_api_restringidas": 7,
        "llamadas_api_sospechosas": 8,
        "permisos_solicitado": 9,
        "permisos_usados": 10,
        "uses_feature": 11
    }

    apk_map = {x: 0 for x in range(1, 12)}
    #print(apk_map)
    # apk_map = {x: 0 for x in range(1, 9)}
    for linea in feature_file:
        if linea != "\n":
            feature_type = linea.split("::")[0]
            print("Linea archivo .txt --> ", linea.split("::"))
            print("feature_type --> ", feature_type)
            apk_map[features_types_array[feature_type]] += 1
            print("features_type[feature_type] --> ", features_types_array[feature_type])
            print("apk_map[1] + 1 --> ", apk_map)
            # Contar características por tipo usando como llave el id del tipo de características
            # apk_total_features_id = {1:2,...,11:12}
            # apk_total_features_id[features_types_array[feature_type]] += 1
            # Contar características por tipo usando como llave el nombre del tipo de características
            # apk_total_feature_name = {"feature":2,...,"url":12}
            apk_total_feature_name[feature_type] = apk_total_feature_name[feature_type] + 1
            # apk_map[features_types_array[feature_type]] = apk_map[features_types_array[feature_type]] + 1
    # print("\n")
    # print(apk_map)
    features = []
    # for i in range(1, 12):
    for i in range(1, 12):
        features.append(apk_map[i])
    # print("\n")
    # print(features)
    return features


def read_sha_files():
    """
    Reads each application file in the directory and uses the function count_feature_set to get the property count for
    each feature and organises it in a mutidimensional array with the filename
    :return: a multi dimensional array containing with each element having the full properties (name and feature set
    count) for each application data file
    """
    # crea la variable contador, que se usará en el print final como número id de la aplicación siendo procesada
    count = 0
    # En este lista se insertará el nombre del apk en formato sha-256 y su vector de frecuencia de características,
    # el cual contiene el número de veces que apareció cada una de los tipos de características.
    # La información de este arreglo se escribirá en el archivo .csv feature_of_counts_temp
    apks_array_csv = []
    # Variable acumuladora de la cantidad de características de las aplicaciones
    features_acum = 0
    # Procesa cada archivo de características de cada aplicación
    # CUANDO DATASET NO ESTÁ BALANCEADO
    for filename in os.listdir(dir_of_files):
        # CUANDO DATASET ESTÁ BALANCEADO
        # for filename in nuevo_dataset:

        # une la ruta donde se encuentran los archivos de características extraídos con el nombre de la aplicación
        # ejemplo: ../raw_data/feature_vectors/0a0a6170880154d5867d59282228f9ed45267b2083cdf6bb351b1ea456ca4105
        # con la ruta completa, puede abrir el archivo donde estan las características
        sha_data = open(dir_of_files + filename)
        # agrega la unión del nombre de la aplicación y el resultado de la función count_feature_set(sha_data)
        # esa función lleva como parámetro la ruta del archivo que está abierta en modo lectura
        # y el resultado de esa función es el arreglo de frecuencia de características, que contiene el número de veces
        # que aparece en la aplicación cada tipo característica
        # ejm: apks_array_csv = ['0a0a6170880154d5867d59282228f9ed45267b2083cdf6bb351b1ea456ca4105',3,8,6,4,11,10,8,21]
        # features_count = count_feature_set(sha_data)
        apks_array_csv.append([filename] + count_feature_set(sha_data))
        # una vez leído ese archivo, se cerrará
        sha_data.close()
        # aumenta en 1, porque el id de cada aplicación es diferente
        count = count + 1

        # imprime datos informativos sobre que aplicación esta siendo procesada
        print("id" + " " + str(count) + " " + "nombre_apk" + " " + filename + " " + "feature vector" + " " +
              str(apks_array_csv[count - 1][1:]))
    # devuelve como resultado el arreglo de la aplicación que contiene su nombre y vector de frecuencia
    print(apk_total_features_id)
    # print("Total de características del dataset: " + str(apk_total_feature_name))
    print("\n")
    print("----- Cantidad de características en el dataset según su tipo -----")
    for feature, feature_count in apk_total_feature_name.items():
        features_acum += feature_count
        print(feature, '-->', feature_count)
    print("--------------------------------------------------------------------")
    #print("\n")
    print("Cantidad total de características del dataset: " + str(features_acum))
    print("--------------------------------------------------------------------")
    print("----- Rerepsentación % del tipo en el dataset -----")
    for feature, feature_count in apk_total_feature_name.items():
        print(feature, '-->', format((feature_count/features_acum), ".2%"))
    print("----------------------------------------------------")
    print("\n")
    return apks_array_csv


def create_csv_for_sha_data():
    """
    Creates a temporary file containing ONLY FEATURE SET INPUTS
    """
    # define los encabezados para el archivo feature_of_counts_temp.csv
    # header = ['sha256', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
    header = ['apkname', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11']
    # abre el archivo feature_of_counts_temp en modo escritura para escribir la cantidad características de cada tipo
    # de los apks
    # with open(feature_of_counts_temp, 'wt', newline='') as file:
    with open(feature_of_counts_temp, "wt", newline='') as file:
        # Usa el objeto de writer para convertir el archivo (file) en cadenas de texto delimitadas por ','
        writer = csv.writer(file, delimiter=',')
        # Escribe el encabezado como primera línea del archivo.
        # writer.writerow(i for i in header)
        writer.writerow(header)
        # Ejecuta la función read_sha_files()
        # su resultado de esa función es el arreglo que contiene el nombre y vector de características de las
        # de las apks
        # ****** Otra forma: ******
        # all_apks_vectors = read_sha_files()
        # for j in all_apk_vectors:
        for j in read_sha_files():
            writer.writerow(j)


def balanceo_dataset():
    """
    Usará el archivo sha_256.csv definida con variable known_malware_file, que contiene los nombres de las aplicaciones
    en formato sha-256, los cuales fueron registrados como malware. la carpeta drebin/feature_vectors definida
    inicialemnte con variable dir_of_files
    :return:
    """
    # Define una lista donde se agregaran las aplicaciones benignas y malware
    non_malware = list()
    malware = list()

    # Lista archivos de características ubicados en la ruta "../raw_data/feature_vectors/" y las almacena en la lista
    # de nombre dataset
    dataset = os.listdir(dir_of_files)

    # Abre el archivo sha256_family.csv
    with open(known_malware_files) as csvfile:
        # Lo comenzará a leer
        reader = csv.reader(csvfile)
        # inicia una iteración
        next(reader)
        # En un nuevo diccionario almacena como llave el nombre de la aplicación y como valor respectivo al nombre de la
        # familia de malware que pertenece
        malware_dictionary = {row[0]: row[1] for row in reader}

    # Recorre la lista dataset que contiene los archivos de características de cada aplicacion
    for i in dataset:
        # Verifica si la aplicacion se encuentra en el diccionario que contiene a los malware
        # Está verificando que aplicaciones de la lista son malware y las agrega a la lista llamada malware
        if i in malware_dictionary:
            malware.append(i)
        # Si la aplicación no está en el diccioonario, entonces la agrega a la lista de aplicaciones benignas
        else:
            non_malware.append(i)
    # Muestra el total de aplicaciones del dataset
    print("***Datos del dataset antes de ser balanceado***")
    print('Size of dataset: ', len(malware) + len(non_malware))
    # Y la cantidad de aplicaciones benignas y malware
    print('Number of non malwares:\t', len(non_malware))
    print('Number of malwares:\t', len(malware))

    # Genera una lista con IDs aleatorios de las aplicaciones benignas. La cantidad de IDs será igual a la cantidad de
    # aplicaciones malware. El rango de valores que puede tomar el ID es de 0 a 123452.
    index = random.sample(range(0, len(non_malware) - 1), len(malware))

    # Usando la anterior lista de índices, agregará las nuevas aplicaciones benignas a la lista
    non_malware = [non_malware[i] for i in index]

    # Mezcla ambas listas (malware y no malware)
    dataset_balanceado = malware + non_malware

    # Vector with class of each example
    # y = [1] * len(malware) + [0] * len(non_malware)

    # Muestra el total de aplicaciones
    print("***Datos del dataset después de ser balanceado***")
    print('Number Total of apps: ', len(dataset_balanceado))

    # Y la cantidad de aplicaciones benignas y malware
    print('Number of non malware: ', len(non_malware))
    print('Number of malware: ', len(malware))
    # retorna la lista donde están las 5560 aplicaciones benignas y 5560 malwares
    return dataset_balanceado


# Balanceo de dataset en 5560 aplicaciones benignas y 5560 malwares
# nuevo_dataset = balanceo_dataset()
# Crea archivo feature_of_counts_temp.csv, el cual será leído luego.
# Este se encuentra ubicado en la ruta ../processed_data/feature_vectors_counts_temp.csv
create_csv_for_sha_data()

"""
    map feature_vectors sha with it's corresponding output value (whether it is a malware or not)
    using the ground truth given in the sha_family file,
"""
# Carga el archivo sha256_family.csv en modo lectura en data
data = pd.read_csv(known_malware_files)
# Del anterior archivo leído escoge y carga los valores de la columna sha256 que tiene 5560 columnas
# Usar con el dataset de Drebin
# malware_column = data["sha256"]
# Usar con cualquier otro dataset excepto el de Drebin
malware_column = data["malware_name"]

# Carga el archivo feature_of_counts_temp.csv en modo lectura
feature_vectors_data = pd.read_csv(feature_of_counts_temp)
# Del anterior archivo selecciona la columna sha 256 que tiene 129013 filas
apk_column = feature_vectors_data['apkname']
# Verfica si cada val   or del arreglo apk_column está en el arreglo malware_column. El primero corresponde al archivo
# generado con el proceso anterior que contiene todas las aplicaciones del dataset. Y el segundo corresponde al archivo
# de aplicaciones que pertenecen a una familia de malware
# Retorna es un arreglo con valores boolean. si el valor de apk_column está, entonces le asigna True, sino False.
# resultado:
#               id          Col0
#               0           False
#               1           False
#               2           True
#               3           True
#               .             .
#               .             .
#               .             .
#               129012      False
mask = np.in1d(apk_column, malware_column)

# creates the full feature vectors file containing both inputs and output (malware or not)
# this file is created as a merger of the temporary file created and the output generated above
# crea un dataframe en base al diccionario con llave 'malware' y valor del arreglo mask
# resultado:
#               id          malware
#               0           False
#               1           False
#               2           True
#               3           True
#               .             .
#               .             .
#               .             .
#               129012      False
malware = pd.DataFrame({'malware': mask})
print(malware)
# Une el dataframe feature_of_counts_temp.csv con el dataframe con una única columna malware. Ambos archivos tienen
# la misma cantidad de columnas. con left_index=True usas el índice del primer dataframe como valor clave y si es
# multi-índice, entonces el índice o número de columnas debe coincidir con el índice del segundo dataframe.
feature_vectors_data = feature_vectors_data.merge(malware, left_index=True, right_index=True)
# Escribe el contenido del anterior dataframe en el archivo feature_of_counts.csv. Este es la unión del archivo
# feature_of_counts_temp.csv con la columna adicional de malware, que indica si la app es malware o no,
# en base al archivo de familias de malware sha256_family.csv
feature_vectors_data.to_csv(feature_of_counts)
