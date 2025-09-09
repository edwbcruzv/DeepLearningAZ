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

dataset = pd.read_csv('Churn_Modelling.csv') # {buscar el dataset}

# Variable independiente
X = dataset.iloc[:,3:-1].values 

# Variable dependiente
y = dataset.iloc[:,[13]].values

# Nota: convertir a matrices tanto a X como a y para evitar problemas
#       al no usar matrices.
# =============================================================================
# --------------------Tratamiendo de NAs--------------------
# =============================================================================

# No aplica ya que el dataset esta completo


# =============================================================================
# --------------------Codificar datos categoricos--------------------
# =============================================================================
# Los datos son categorias que se deben de tranformar a numeros para
# que python los pueda trabajar.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# la funcion se encargara de transformar las categorias a datos numericos.
le_X1 =LabelEncoder()
le_X2 =LabelEncoder()

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

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)

# =============================================================================
# --------------------Escalado de variables--(obligatorio cuando son RNs)
# =============================================================================

from sklearn.preprocessing import StandardScaler

# Escalador para las variables independientes
sc_X = StandardScaler()
# escalando variables de training, se usa el fit_trasform
X_train = sc_X.fit_transform(X_train)

# Se escala con el mismo escalador con las variables de testing con transform
# para que la trasformacion lo haga en base al conjunto escalado de training
X_test = sc_X.transform(X_test)


# =============================================================================
# PARTE 2:  Construyendo la Red Neuronal Artificial
# =============================================================================
# =============================================================================
# Importar keras y librerias adicionales
# =============================================================================
import keras 
from keras.models import Sequential # Inicializa los parametros de la RNA
from keras.layers import Dense # Crear las conexiones entre capas de la RNA y asignador de pesos
from keras.layers import Dropout # para evitar el overfitting
# =============================================================================
# Inicializar la Red Neuronal Artificial
# =============================================================================
# INICIALIZAR LA RED NEURONAL 
classifier=Sequential()
# =============================================================================
# INTRODUCIR LA PRIMERA OBSERVACION DEL DATASET A LA CAPA DE ENTRADA
# (primera capa oculta)
# No. de nodos sera el promedio del los numero de datos de entrada y 
# el de salida, en este caso: Entrada 11, Salida 1. la media es 6
# =============================================================================
classifier.add(Dense(units=6,# sinapsis (salida)
                     # Funcion de distribucion de los pesos de entrada
                     kernel_initializer='uniform',
                     # Funcion de activacion
                     activation='relu',
                     # Dimension de entrada (11 datos de entrada)
                     input_dim=11))
# classifier.add(Dropout(p=0.1))
# =============================================================================
# Añadir la segunda capa oculta
# =============================================================================
classifier.add(Dense(units=6,# sinapsis (salida)
                     # Funcion de distribucion de los pesos de entrada
                     kernel_initializer='uniform',
                     # Funcion de activacion
                     activation='relu',
                     # Dimension de entrada no se define porque sabe cuantos recibe
                     ))
# classifier.add(Dropout(p=0.1))
# =============================================================================
# Añadir la ultima capa (capa de salida)
# =============================================================================
classifier.add(Dense(units=1,# sinapsis (la salida de la red)
                     # Funcion de distribucion de los pesos de entrada
                     kernel_initializer='uniform',
                     # Funcion de activacion
                     activation='sigmoid',
                     # Dimension de entrada no se define porque sabe cuantos recibe
                     ))
# =============================================================================
# Compilar la Red Neuronal Artificial
# =============================================================================
classifier.compile(optimizer="adam", # optimizador
                   loss='binary_crossentropy', # funcion de perdidas, minimixando el error
                   metrics=['accuracy'] # metrica de precision
                   )
# =============================================================================
# Ajustamos la Red Neuronal Artificial al conjunto de entrenamiento
# =============================================================================
classifier.fit(X_train,y_train,
               batch_size=10, # Numero de bloques a procesar antes de ajustar los pesos,(evita overfiting)
               epochs=100) # Numero de iteraciones globales sobre el conjunto
# =============================================================================
# PARTE 3:  Evaluar el modelo y calcular predicciones finales
# =============================================================================
# =============================================================================
# Prediccion de los resultados con el conjunto de testing
# =============================================================================

y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5) # umbral

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
porcent=(c_m[0][0]+c_m[1][1])/c_m.sum()

# =============================================================================
# PARTE 3: Evaluar el modelo y calcular predicciones finales
# =============================================================================
# =============================================================================
# Nueva prediccion (Tarea)
# =============================================================================


new_predict=classifier.predict(
    sc_X.transform(
        np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))

print(new_predict>0.5)

# =============================================================================
# Evaluar la RNA con K-Fold
# =============================================================================
# from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    
    classifier=Sequential()
    # ========================================================================
    # Añadir las capas de entrada y primera capa oculta
    # ========================================================================
    classifier.add(Dense(units=6,# sinapsis(media)
                         kernel_initializer='uniform', # funcion de distribucion
                         activation='relu',# Funcion de activacion
                         input_dim=11))
    # ========================================================================
    # Añadir la segunda capa oculta
    # ========================================================================
    classifier.add(Dense(units=6,# sinapsis(media)
                         kernel_initializer='uniform', # funcion de distribucion
                         activation='relu',# Funcion de activacion
                         ))
    # ========================================================================
    # Añadir la ultima capa (capa de salida)
    # ========================================================================
    classifier.add(Dense(units=1,# sinapsis(media)
                         kernel_initializer='uniform', # funcion de distribucion
                         activation='sigmoid',# Funcion de activacion
                         ))
    # ========================================================================
    # Compilar la Red Neuronal Artificial
    # ========================================================================
    classifier.compile(optimizer="adam", # optimizador
                       loss='binary_crossentropy', # perdida
                       metrics=['accuracy'] # metrica de precision
                       )
    return classifier

# similar al fit
classifier2=KerasClassifier(build_fn=build_classifier, # obtenemos la red neuronal fabricada
                            batch_size=10, #tamaños de bloque
                            epochs=100) # epocas
# trabajara dividiendo los bloques obteniendo las presisiones para obtener la media y varianza
accuracies = cross_val_score(estimator=classifier2, # objeto para ajustar
                             X=X_train,
                             y=y_train,
                             cv=10, # despliegues, la 10 e sla estandar
                             verbose=1, # muestre el procesos
                             n_jobs=-1) # para desplegar los nucleos del cpu, si es -1 se utilizara toda la potencia disponible

mean=accuracies.mean() # promedio 
# mean=0.84799999
variance=accuracies.std() # desviacion estandar
# variance= 0.0127867070
# =============================================================================
# Mejora la RNA
# =============================================================================
# Regularizacion del dropou para evitar overfitting,
# en este caso no decidi ajustarlo, ya que tenemos un 85 % de coincidencias
# en las predicciones
# Dichos parametros estan comentados en las capas ocultas de la RNA
# =============================================================================
# Ajustar la RNA (probar despues ya que tarda mucho y la libreria esta dando problemas)
# =============================================================================
from sklearn.model_selection import GridSearchCV

# Mandaremos como argumento el optimizador
def build_classifier2(optimizer):
    
    classifier=Sequential()
    # ========================================================================
    # Añadir las capas de entrada y primera capa oculta
    # ========================================================================
    classifier.add(Dense(units=6,# sinapsis(media)
                         kernel_initializer='uniform', # funcion de distribucion
                         activation='relu',# Funcion de activacion
                         input_dim=11))
    # ========================================================================
    # Añadir la segunda capa oculta
    # ========================================================================
    classifier.add(Dense(units=6,# sinapsis(media)
                         kernel_initializer='uniform', # funcion de distribucion
                         activation='relu',# Funcion de activacion
                         ))
    # ========================================================================
    # Añadir la ultima capa (capa de salida)
    # ========================================================================
    classifier.add(Dense(units=1,# sinapsis(media)
                         kernel_initializer='uniform', # funcion de distribucion
                         activation='sigmoid',# Funcion de activacion
                         ))
    # ========================================================================
    # Compilar la Red Neuronal Artificial
    # ========================================================================
    classifier.compile(optimizer=optimizer, # optimizador
                       loss='binary_crossentropy', # perdida
                       metrics=['accuracy'] # metrica de precision
                       )
    return classifier

classifier3=KerasClassifier(build_fn=build_classifier2)

# Hiperparametros que queremos optimizar
parameters={
    'batch_size':[25,32],# parametro a ajustar: el rango de valores que queremos que se pruebe y busque el optimo
    'epochs':[100,500],
    'optimizer':['adam','rmsprop']
    }

grid_search=GridSearchCV(estimator=classifier3,
                         param_grid=parameters,
                         scoring='accuracy',
                         cv=10)

grid_search=grid_search.fit(X_train,y_train)



# Resultados en su momento
best_parameters=grid_search.best_params_ # mejores parametros 
# Ejemplo: best_parameters={'batch_size':25,'epochs':500,'optimizer':'adam'}
best_accuracy=grid_search.best_score_ # la mejor presicion obtenida   
# Ejemplo: best_accuracy=0.8515


