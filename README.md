# Prueba-de-concepto Seminario de Investigación II #
Este repositorio contiene el código fuente de mi metodología propuesta está compuesto de 3 etapas y basada en 2 
metodologías de 2 autores.

## Metodologías tomadas como base: ##
<Blockquote> Sayo Makinwa. Drebin-malware-analysis. https://github.com/sayomakinwa/drebin-malware-analysis </Blockquote>
<Blockquote> Annamalai Narayanan. Drebin. https://github.com/MLDroid/drebin </Blockquote>

## Etapas: ##
1. Descarga del dataset de aplicaciones
2. Extracción de tipos de atributos
3. Selección de características
4. Modelo de clasificación

#### Observaciones importantes antes de ejecutar el código ####
* El repositorio tiene dos carpetas de entorno virtual: uno de Python 2.7 y otro de Python 3. El primero se usa en la 
Extracción de características y el segundo en para Selección de características y el Modelo de clasificación.
* Estos entornos ya contienen las librerías necesarias para ejecutar el código; por lo tanto, solo falta activarlos para
ser usados.

Usando la línea de comandos ubícate en el directorio donde está la carpeta del entorno virtual y actívalo con el comando
<code>  source venv-2.7/bin/activate </code>

En este ejemplo, se activó el entorno virtual de python 2.7 para usar en la primera etapa.
Para activar el entorno virtual de python 3, el cual se usa en la segunda y tercera etapa, se usa un comando simmilar
<code>  source venv-3.8/bin/activate </code>

Una vez activado el entorno virtual, continúa con la instalación de las librerías.

### 1ra. Etapa: Extracción de tipos de atributo ###
#### Requerimientos de ejecución: ####
El código está basado en Python 2.7 y fue testeado usando el IDE PyCharm (versión 2020.1.2 RC) con un entorno virtual 
de Python 2.7, y en una PC con sistema operativo Ubuntu 20.04.
Se recomienda usar un entorno virtual, ya sea en el IDE de Pycharm u otro, o creándolo desde la terminal de comandos e 
instalar las siguientes librerías:
<code> 
* Pebble (4.5.3)
* glob2 (0.7)
* joblib (0.14.1)
* networkx (2.2)
* numpy (1.16.6)
* psutil (5.7.0)
* scikit-learn (0.20.4)
</code>

#### Observaciones importantes ####
* Las versión indicada en cada librería fue usada para ejecutar el código de esta etapa.
* En el caso de las librerías <code> joblib, networkx, numpy y scikit-learn </code> existe una versión más actual; 
sin embargo no son compatibles en python 2, sino con python 3.

#### Pasos de ejecución: ####
##### Usando la línea de comandos #####
i. Clona el repositorio a un directorio de tu PC.

ii. Ubicate en la ruta <code> extraccion_tipos_atributos/code/ </code>.

iii. Crea un entorno virtual de python 2.7 en la carpeta de tu elección y actívalo.

iv. Instala las librerías usando el comando <code>pip2 install nombre_libreria </code>.

v. Ejecuta el comando <code> python Main.py --help </code> para ver los parámetros de entrada.

Estos son:

<code> --mw_dir '../data/apks/malware' </code> --> Ruta del directorio que contiene las aplicaciones (.apks) malware. 
La ruta por defecto es <code>'../data/apks/malware'</code>

<br>

<code> --gw_dir '../data/apks/goodware' </code> --> Ruta del directorio que contiene las aplicaciones (.apks) benignas. 
La ruta por defecto es <code>'../data/apks/goodware'</code>

<br>

<code> --cpucores CPUCORES </code>  Máximo número de núcleos del CPU que se usarán para el multiprocesamiento de las apks (parámetro 
  usado en la extracción de tipos de atributo). Por defecto, usará el máximo número de nucleos identificado de tu PC.

vi. Los archivos .txt que contienen las características de las aplicaciones son los archivos de salida de esta etapa y 
están en <code>data/apks/malware/vectores_txt</code> y <code>data/apks/goodware/vectores_txt</code>

##### Usando la IDE Pycharm Community #####
i. Clona el repositorio a un directorio de tu PC

ii. Abre la carpeta del repositorio como un nuevo proyecto en Pycharm.

iii. Crea y configura el entorno virtual de python 2.7 en la carpeta de tu elección.

iv. Instala las librerías, ya sea usando la terminal de comandos de Pycharm o desde la configuración del entorno virtual.

v. Configura los parámetros a usar. Estos son los mismos indicados arriba.

vi. Ejecuta el archivo Main.py.

vii. Los archivos .txt que contienen las características de las aplicaciones son los archivos de salidas de esta etapa y
se ubican en <code>data/apks/malware/vectores_txt</code> y <code>data/apks/goodware/vectores_txt</code>.

### 2da. Etapa: Seleccón de características ###
#### Requerimientos de ejecución: ####
El código está basado en Python 3 y fue testeado usando el IDE PyCharm (versión 2020.1.2 RC) con un entorno virtual 
de Python 3.8, y en una PC con sistema operativo Ubuntu 20.04.
Se recomienda usar un entorno virtual, ya sea en el IDE de Pycharm u otro, o creándolo desde línea de comandos e instalar 
la siguiente librería:

<code> 

* Pandas (1.0.4) 
</code>

#### Pasos de ejecución: ####
##### Usando la línea de comandos #####
i. Una vez hayas clonado el repositorio, ubícate en la carpeta <code> seleccion_caracteristicas/code/ </code>.

ii. Crea un entorno virtual de python 3 en la carpeta de tu elección. Este entorno virtual será reutilizado por el 
código de la tercera etapa.

iii. Activa el entorno virtual.

iv. Instala la librería usando el comando <code> pip3 install nombre_libreria </code>.

v.  Crea la carpeta <code>processed_data</code> y el directorio <code>/raw_data/feature_vectors</code> dentro de <code>seleccion_caracterisicas</code>.

vi. Ejecuta el archivo <code> data_vectorization.py </code> para comenzar la segunda etapa.

vii. Los archivos de entrada en esta etapa está ubicados en <code>/raw_data/feature_vectors</code>. Estos archivos .txt 
son los archivos de salida de la primera etapa y contienen las características extraídas de las aplicaciones.

viii. El archivo de salida de esta etapa es feature_vectors_counts.csv; está compuesto de 12 columnas: la primera 
corresponde al nombre de la aplicación; y las 11 restantes a los 11 tipos de atributos extraídos. Este archivo muestra 
la cantidad de veces que aparece cada tipo de atributo en cada aplicación del dataset.

##### Usando la IDE Pycharm Community #####
i. Una vez hayas clonado el repositorio y hayas abierto la carpeta del mismo como un nuevo proyecto en Pycharm, crea y 
configura el entorno virtual de python 3 en la carpeta de tu elección.

ii. Instala la librería, ya sea usando la terminal de comandos de Pycharm o desde la configuración del entorno virtual.

iii. Crea la carpeta <code>processed_data</code> y el directorio <code>/raw_data/feature_vectors</code> dentro de <code>seleccion_caracterisicas</code>.

iv. Ejecuta el archivo <code> data_vectorization.py </code>.

v. Los archivos de entrada en esta etapa está ubicados en <code>/raw_data/feature_vectors</code>. Estos archivos .txt 
son los archivos de salida de la primera etapa y contienen las características extraídas de las aplicaciones.

vi. El archivo de salida de esta etapa es feature_vectors_counts.csv; está compuesto de 12 columnas: la primera corresponde 
al nombre de la aplicación; y las 11 restantes a los 11 tipos de atributos extraídos. Este archivo muestra la cantidad 
de veces que aparece cada tipo de atributo en cada aplicación del dataset.


### 3da. Etapa: Modelo de clasificación ###
#### Requerimientos de ejecución: ####
El código también está basado en Python 3 y fue testeado usando el IDE PyCharm (versión 2020.1.2 RC) con un entorno virtual 
de python 3.8, y en una PC con sistema operativo Ubuntu 20.04.
Se recomienda usar un entorno virtual, ya sea en el IDE de Pycharm u otro, o creándolo desde línea de comandos, e 
instalar las siguientes librerías:

<code> 

* pandas (1.0.4)
* scikit-learn (0.23.1)
* seaborn (0.3.1)
</code>

#### Pasos de ejecución: ####
##### Usando la línea de comandos #####
i. Una vez hayas clonado el repositorio, ubícate en la carpeta <code> modelo_clasificacion/code/ </code>.

ii. Crea un entorno virtual de python 3 en la carpeta de tu elección o reutiliza el entorno virtual de python 3 creado para la 
segunda etapa.

iii. Activa el entorno virtual.

iv. Instala las librerías usando el comando <code> pip3 install nombre_libreria </code>.

v. Ejecuta el archivo <code> rfc.py </code> para comenzar la tercera etapa.

vi. El archivo de entrada feature_vectors_counts.csv se ubica en el directorio <code>seleccion_caracteristicas/processed_data</code>.

vii. En esta etapa la salida son los resultados de las métricas de Accuracy, Precision, Recall y F1 Score calculadas, 
además de la Matriz de confusión.

##### Usando la IDE Pycharm Community #####
i. Una vez hayas clonado el repositorio y hayas abierto la carpeta del mismo como un nuevo proyecto en Pycharm, crea y 
configura el entorno virtual de python 3 en la carpeta de tu elección. O reutiliza el entorno virtual de python 3 creado 
y configurado en la anterior etapa.

ii. Instala las librerías, ya sea usando la terminal de comandos de Pycharm o desde la configuración del entorno virtual.

iv. Ejecuta el archivo <code> rfc.py </code>.

v. El archivo de entrada feature_vectors_counts.csv se ubica en el directorio <code>seleccion_caracteristicas/processed_data</code>.

vi. En esta etapa la salida son los resultados de las métricas de Accuracy, Precision, Recall y F1 Score calculadas, 
además de la Matriz de confusión.
