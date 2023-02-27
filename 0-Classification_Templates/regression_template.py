# -*- coding: utf-8 -*-
"""
Date: Sat Feb 11 16:48:36 2023

@author: edwin
"""

# =============================================================================
# Plantilla de Regresion Lineal Polinomica
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

# Estructura de los datos: {explicar eldataset y el objetivo}.
# Filas :{numero de filas}
# Columnas:
#           |{col1}|{col2}|{...} (vars independiente)
#           |{columna de var indep.}| (var_dependiente)

dataset = pd.read_csv('dataset.csv') # {buscar el dataset}

# Variable independiente:Mayuscula por ser una matriz.
#   tomamos [Todas las filas ,Solo la columna(s)...]
X = dataset.iloc[:,1:2].values # {se pueden modificar segun se necesite}

# Variable dependiente:minuscula por ser un vector.
#   tomamos [Todas las filas: Solo la ultima columna]
y = dataset.iloc[:,2:3].values # {se pueden modificar segun se necesite}

# =============================================================================
# --------------------Dividiendo dataset en conjuntos--------------------
# --------------------de entrenamiento y conjunto de testings-------------
# =============================================================================
# {se pueden modificar segun se necesite}
# =============================================================================
# Ajustar la regresion {sea cualquier tipo} con el dataset
# =============================================================================

# from sklearn.preprocessing import TypeRegression
# type_regression=TypeRegression(degree=4) # se puede jugar con el grado
# X_reg=type_regression.fit_transform(X)

# =============================================================================
# Prediccion de nuestros modelos (Resultados)
# =============================================================================
y_pred=regression.predict()

# =============================================================================
# Visualizacion de los resultado: Modelo Polinomico
# =============================================================================
plt.scatter(X,y,color='red')
plt.plot(X,regression.predict(),color='green')
plt.title("Modelo Regresion {type}")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo en $")
plt.show()
















