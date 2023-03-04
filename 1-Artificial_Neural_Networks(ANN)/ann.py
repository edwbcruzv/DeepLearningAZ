# -*- coding: utf-8 -*-
"""
Date: Wed Feb 22 20:15:46 2023

@author: edwin
"""
# =============================================================================
# Redes Neuronales Artificiales
# =============================================================================
# =============================================================================
# --------------------Importando librerias--------------------
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# =============================================================================
# --------------------Importando dataset--------------------
# =============================================================================

# Estructura de los datos: Clientes de un banco.
# Objetivo: analizar si despues de 6 meses un cliente se queda en el banco.
# Filas :{numero de filas}
# Columnas:
#           |{...}| (vars independiente)
#           |Exited| (var_dependiente)

dataset = pd.read_csv('Churn_Modelling.csv') # {buscar el dataset}

cliente1=np.asarray([[0,0,600,1,40,3,6000,2,1,1,50000]])

# Variable independiente:Mayuscula por ser una matriz.
#   tomamos [Todas las filas ,desde la 4ta columna hasta la penultima]
# el resto son datos irrelevantes
X = dataset.iloc[:,3:-1].values 

# Variable dependiente:minuscula por ser un vector.
#   tomamos [Todas las filas: Solo la ultima columna]
y = dataset.iloc[:,[13]].values

# Nota: convertir a matrices tanto a X como a y para evitar problemas
#       al no usar matrices.
# =============================================================================
# --------------------Tratamiendo de NAs--------------------
# =============================================================================

# from sklearn.impute import SimpleImputer
# Los valores desconocidos de los valores independientes son los NA´s.
# El valor que se va a sustituir que sera la media.
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Seleccionando las columnas las cuales estan los valores NA´s.
# [Todas las filas,Columnas 1 y 2]
# imputer = imputer.fit(X[:,1:3]) # ajustando valores
# Sobreescribirnedo la matriz con la nueva trasformacion configurada.
# X[:,1:3] = imputer.transform(X[:,1:3])


# =============================================================================
# --------------------Codificar datos categoricos--------------------
# =============================================================================
# Los datos son categorias que se deben de tranformar a numeros para
# que python los pueda trabajar.
# En este caso las columnas "Geography" y "Gender".
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# la funcion se encargara de transformar las categorias a datos numericos.
le_X1 =LabelEncoder()
le_X2 =LabelEncoder()
# De la tabla de variables independientes se toma 
# la columna "Country"y todas las filas. Y se sobreescribe la tabla.
X[:,1]=le_X1.fit_transform(X[:,1])
X[:,2]=le_X2.fit_transform(X[:,2])

# Ahora se debe de transformar la columnas categoricas a variables dummy,
# creando una columna por cada categoria y de las variables dummy solo se
# marca la categoria correcta con un booleano.


ct = ColumnTransformer(
    # Lista de tuplas (nombre,transformador,columnas) que se le aplicara 
    # al conjunto de datos.
    [('one_hot_encoder',OneHotEncoder(categories='auto',dtype=int),[1])],
    # Se pasa el resto de columnas que no se tocaron.
    remainder='passthrough')

X = np.array(ct.fit_transform(X),dtype=float)
X=X[:,1:] # Se elimina la columna de Francia (multicolinealidad)



# =============================================================================
# --------------------Dividiendo dataset en conjuntos--------------------
# --------------------de entrenamiento y conjunto de testings--------------------
# =============================================================================

from sklearn.model_selection import train_test_split
# la sig funcion devolvera varias variables con los valores de testing y training
# Como parametros:Matriz independiente,
#           matridependiente a predecir,
#           tamaño del conjunto de testing en % (el resto se va a entrenamiento),
#           numero random de division de datos (semilla random=cualquier numero).
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

# =============================================================================
# --------------------Escalado de variables--------------------
# =============================================================================

from sklearn.preprocessing import StandardScaler

# Escalador para las variables independientes
sc_X = StandardScaler()

sc_client=StandardScaler()
# escalando variables de training, se usa el fit_trasform
X_train = sc_X.fit_transform(X_train)

cliente1= sc_client.fit_transform(cliente1)
# Se escala con el mismo escalador con las variables de testing con transform
# para que la trasformacion lo haga en base al conjunto escalado de training
X_test = sc_X.transform(X_test)

## En este caso no es necesario escalar las variables dependiente,
## pero en otras ocaciones si se necesitaran escalar

# =============================================================================
# PARTE 2:  Construyendo la Red Neuronal Artificial
# =============================================================================
# =============================================================================
# Importar keras y librerias adicionales
# =============================================================================
import keras 
from keras.models import Sequential
from keras.layers import Dense
# =============================================================================
# Inicializar la Red NEuronal Artificial
# =============================================================================
classifier=Sequential()
# =============================================================================
# Añadir las capas de entrada y primera capa oculta
# =============================================================================
classifier.add(Dense(units=6,# sinapsis(media)
                     kernel_initializer='uniform', # funcion de distribucion
                     activation='relu',# Funcion de activacion
                     input_dim=11))
# =============================================================================
# Añadir la segunda capa oculta
# =============================================================================
classifier.add(Dense(units=6,# sinapsis(media)
                     kernel_initializer='uniform', # funcion de distribucion
                     activation='relu',# Funcion de activacion
                     ))
# =============================================================================
# Añadir la ultima capa (capa de salida)
# =============================================================================
classifier.add(Dense(units=1,# sinapsis(media)
                     kernel_initializer='uniform', # funcion de distribucion
                     activation='sigmoid',# Funcion de activacion
                     ))
# =============================================================================
# Compilar la Red Neuronal Artificial
# =============================================================================
classifier.compile(optimizer="adam", # optimizador
                   loss='binary_crossentropy', # perdida
                   metrics=['accuracy'] # metrica de precision
                   )
# =============================================================================
# Ajustamos la Red Neuronal Artificial al conjunto de entrenamiento
# =============================================================================
classifier.fit(X_train,y_train,batch_size=10,epochs=100)
# =============================================================================
# PARTE 3:  Evaluar el modelo y calcular predicciones finales
# =============================================================================
# =============================================================================
# Prediccion de los resultados con el conjunto de testing
# =============================================================================

y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)

cliente1_pred=classifier.predict(cliente1)

y_test=y_test.astype(bool)

# =============================================================================
# Elaborar una Matriz de confusion
# 
# |----------------------|----------------------|
# |     Verdaderos       |      Falsos          |
# |     Positivo         |      Positivos       |
# |----------------------|----------------------|
# |     Falsos           |      Verdaderos      |
# |     Negativos        |      Negativos       |
# |----------------------|----------------------|
# =============================================================================

from sklearn.metrics import confusion_matrix

c_m=confusion_matrix(y_test, y_pred)
