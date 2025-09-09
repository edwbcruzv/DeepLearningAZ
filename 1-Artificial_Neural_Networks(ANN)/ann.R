 # =============================================================================
# Redes Neuronales Artificiales
# =============================================================================
# =============================================================================
# --------------------Importando librerias--------------------
# =============================================================================

# =============================================================================
# --------------------Importando dataset--------------------
# =============================================================================

# Estructura de los datos: Clientes de un banco.
# Objetivo: analizar si despues de 6 meses un cliente se queda en el banco.

dataset = read.csv('Churn_Modelling.csv') # {buscar el dataset}

# Variable independiente
dataset = dataset[,4:14]
# Codificando los factores para la RNA
dataset$Geography = as.numeric (factor(dataset$Geography,
                           # dandole un valos a cada etiqueta dentro de la columna
                           levels = c("France","Spain","Germany"),
                           # etiquetas
                           labels = c(1,2,3)))

dataset$Gender = as.numeric (factor(dataset$Gender,
                           levels = c("Female","Male"),
                           # etiquetas
                           labels = c(1,2)))

# =============================================================================
# --------------------Tratamiendo de NAs--------------------
# =============================================================================
# No aplica ya que el dataset esta completo
# =============================================================================
# --------------------Codificar datos categoricos--------------------
# =============================================================================
# No aplica
# =============================================================================
# --------------------Dividiendo dataset en conjuntos--------------------
# --------------------de entrenamiento y conjunto de testings--------------------
# =============================================================================

# install.packages("caTools") # solo se necesita ejecutar una vez
library(caTools)

# configurando semilla aleatoria para la division de datos
set.seed(123)
# se elige el porcentaje de los datos para el training en %
split = sample.split(dataset$Exited,SplitRatio = 0.8)
print(split)
# Dividiendo el conjunto , False para el test
training_set = subset(dataset,split == TRUE)
# Dividiendo el conjunto , True para el training
testing_set = subset(dataset,split == FALSE)

# =============================================================================
# --------------------Escalado de variables--(obligatorio cuando son RNs)
# =============================================================================
training_set[,-11] = scale(training_set[,-11])
testing_set[,-11] = scale(testing_set[,-11])

# =============================================================================
# PARTE 2:  Construyendo la Red Neuronal Artificial
# =============================================================================
# =============================================================================
# Importar librerias adicionales
# =============================================================================
library(h2o)
# =============================================================================
# Inicializar la Red Neuronal Artificial
# =============================================================================
h2o.init(nthreads = -1)


classifier = h2o.deeplearning(
  y = "Exited",
  training_frame = as.h2o(training_set),
  activation = "Rectifier",
  hidden = c(6,6),
  epochs = 100,
  train_samples_per_iteration = -2
)

# =============================================================================
# PARTE 3:  Evaluar el modelo y calcular predicciones finales
# =============================================================================
# =============================================================================
# Prediccion de los resultados con el conjunto de testing
# =============================================================================
prob_pred = h2o.predict(classifier,  newdata = as.h2o(testing_set[,-11]))

y_pred = (prob_pred > 0.5 )

y_pred = as.vector(y_pred)
print(y_pred)
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

cm = table(testing_set[,11],y_pred)
print(cm)
# =============================================================================
# PARTE 3: Evaluar el modelo y calcular predicciones finales
# =============================================================================
# =============================================================================
# Nueva prediccion (Tarea) Arreglar
# =============================================================================


new_predict=classifier.predict(
  sc_X.transform(
    np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))

print(new_predict>0.5)


# =============================================================================
# Cerrar sesion de h2o
# =============================================================================
h2o.shutdown()

