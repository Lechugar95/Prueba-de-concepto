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


### 1ra. Etapa: Extracción de tipos de atributo ###
#### Requerimientos de ejecución: ####
El código está basado en Python 2.7 y fue testeado usando el IDE PyCharm (versión 2020.1.2 RC) con un entorno virtual 
de Python 2.7, y en una PC con sistema operativo Ubuntu 20.04.
Se recomienda usar un entorno virtual, ya sea en el IDE de Pycharm u otro, e instalar las siguientes librerías:
<code> 
* Pebble (4.5.3)
* glob2 (0.7)
* joblib (0.14.1)
* networkx (2.2)
* numpy (1.16.6)
* psutil (5.7.0)
* scikit-learn (0.20.4)
</code>

#### Importantes observaciones: ####
* Las versión indicada de cada librería fue la usada para ejecutar esta el código de esta 1ra. etapa. 
* En casos de las librerías de <code> joblib, networkx, numpy y scikit-learn </code> existe una versión de librería más actual; sin embargo
esas versiones actuales son usadas en python 3 y no son compatibles en python 2.

#### Pasos de ejecución: ####

##### Usando la línea de comandos #####

i. Clona el repositorio a un directorio de tu PC

ii. Ubicate en la ruta <code> extraccion_tipos_atributos/code/ </code>

ii. Crea un entorno virtual de python 2.7 en la carpeta de tu elección

iii. Activa el entorno virtual

iv. Instala las librerías usando el comando <code> pip install nombre_libreria </code>

v.  Ejecuta el comando <code> python Main.py --help </code> para ver los parámetros de entrada

Estos son:

<code> --mw_dir '../data/apks/malware' </code> --> Ruta del directorio que contiene las aplicaciones (.apks) malware. 
La ruta por defecto es <code>'../data/apks/malware'</code>

<br>

<code> --gw_dir '../data/apks/goodware' </code> --> Ruta del directorio que contiene las aplicaciones (.apks) benignas. 
La ruta por defecto es <code>'../data/apks/goodware'</code>

<br>

<code> --cpucores CPUCORES </code>  Máximo número de núcleos del CPU que se usarán para el multiprocesamiento de las apks (parámetro 
  usado en la extracción de tipos de atributo). Por defecto, usará el máximo número de nucleos identificado de tu PC.

vi. Los archivos .txt con las características de las aplicaciones (salida) están dentro de las carpetas 
<code>data/apks/malware/vectores_txt</code> y <code>data/apks/goodware/vectores_txt</code>

##### Usando la IDE Pycharm Community #####
i. Clona el repositorio a un directorio de tu PC

ii. Abre la carpeta del repositorio como un nuevo proyecto en Pycharm

iii. Crea y configura el entorno virtual de python 2.7 en la carpeta de tu elección

iv. Instala las librerías, ya sea usando la terminal de comandos de Pycharm o desde la configuración del entorno virtual.

v. Configura los parámetros a usar. Estos son los mismos indicados arriba.

v.  Ejecuta el archivo Main.py.

vi. Los archivos también se ubican en ambas rutas mencionadas arriba.

### 2da. Etapa: Seleccón de características ###
#### Requerimientos de ejecución: ####
El código está basado en Python 3 y fue testeado usando el IDE PyCharm (versión 2020.1.2 RC) con un entorno virtual 
de Python 3.8, y en una PC con sistema operativo Ubuntu 20.04.
Se recomienda usar un entorno virtual, ya sea en el IDE de Pycharm u otro, e instalar las siguientes librerías:

<code> 

* Pebble (4.5.3)
</code>

#### Pasos de ejecución: ####

##### Usando la línea de comandos #####

i. Clona el repositorio a un directorio de tu PC

ii. Ubícate en la ruta <code> seleccion_caracteristicas/code/ </code>

ii. Crea un entorno virtual de python 3 en la carpeta de tu elección. Este entorno virtual será reutilizado por el 
código de la tercera etapa.

iii. Activa el entorno virtual.

iv. Instala la librería usando el comando <code> pip install nombre_libreria </code>

v.  Crear la carpeta <code>/processed_data</code> y el directorio <code>/raw_data/feature_vectors</code> dentro de la 
carpeta <code>seleccion_caracterisicas</code>

vi. Ejecuta el archivo <code> data_vectorization.py</code> para comenzar la segunda etapa.

vi. Los archivos .txt con las características de las aplicaciones tanto malware y benignas (archivos de entrada), los 
cuales fueron creados en  la primera etapa, son datos de entrada para esta segunda etapa y están ubicados en la carpeta  <code>/raw_data/feature_vectors</code>.

##### Usando la IDE Pycharm Community #####





