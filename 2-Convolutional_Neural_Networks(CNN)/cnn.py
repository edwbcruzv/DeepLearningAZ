# =============================================================================
# Redes Neuronales Artificiales Convolucionales
# =============================================================================
# =============================================================================
# Constuccion del modelo de CNN
# =============================================================================
# Importacion de librerias
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# =============================================================================
# Inicializando la CNN
# =============================================================================
classifier = Sequential()
# =============================================================================
# PASO 1: Convolucion
# =============================================================================
classifier.add(Conv2D(filters=32, # num detectores de caracteristicas
                             kernel_size=(3,3), # tamaño del filtro
                             input_shape= (64,64,3), # tamaño de las imagenes y los 3 canales de color
                             activation="relu") )
# =============================================================================
# PASO 2: Max Pooling
# =============================================================================
classifier.add(MaxPooling2D(pool_size=(2,2))) # define el tamaño del maxpoling
# =============================================================================
# Segunda capa de convolucion y maspoling 
# (mejora) (se puede comentar para notar las diferencias al entrenar)
# =============================================================================
# classifier.add(Conv2D(filters=32, kernel_size=(3,3), activation="relu") )
# classifier.add(MaxPooling2D(pool_size=(2,2)))
# =============================================================================
# PASO 3: Flattening
# =============================================================================
classifier.add(Flatten()) # convierte la amtriz a un vector unidimensional
# =============================================================================
# PASO 4: Full Connection
# =============================================================================
classifier.add(Dense(units=128,  # Capa oculta
                     activation="relu"))
classifier.add(Dense(units=1, # capa de salida
                     activation="sigmoid"))
# =============================================================================
# Compilando la red neuronal de convolucion
# =============================================================================
classifier.compile(optimizer="adam",  
                   loss="binary_crossentropy", 
                   metrics=["accuracy"])

# =============================================================================
# Ajustar la CNN a las imagenes para entrenar
# =============================================================================

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Crear un generador con aumentación/transfomacion de datos
train_datagen = ImageDataGenerator(
    shear_range=0.2,        # Transformación de corte
    zoom_range=0.2,         # Zoom aleatorio
    horizontal_flip=True,   # Inversión horizontal
    rescale=1./255          # Normalización (valores entre 0 y 1)
)

# solo reescalamos en el conjunto de test
test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar imágenes de entrenamiento desde un directorio
train_generator = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64), # Redimensionar imágenes a 64x64
    batch_size=32,          # Tamaño del lote
    class_mode='binary' # Para clasificación multiclase
)

# Cargar imágenes de test desde un directorio
test_generator = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64), # Redimensionar imágenes a 64x64
    batch_size=32,          # Tamaño del lote
    class_mode='binary' # Para clasificación multiclase
)

# Esto puede tardar unos 20 a 30 mis1

classifier.fit(
    train_generator,
    steps_per_epoch=8000,
    epochs=25,
    validation_data=test_generator,
    validation_steps=2000
)

# no dice que valor es cada clase
train_generator.class_indices
test_generator.class_indices

# =============================================================================
# Guardar el modelo entrenado (incluyendo la arquitectura y los pesos)
# =============================================================================
classifier.save('modelo_entrenado.h5')
# =============================================================================
# Cargar el modelo
# =============================================================================
from tensorflow.keras.models import load_model

classifier = load_model('modelo_entrenado.h5')
# =============================================================================
# Predicciones
# =============================================================================
from tensorflow.keras.preprocessing import image
import numpy as np

# Cargar y preprocesar la primera imagen
img1 = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64, 64))
img1_array = image.img_to_array(img1) / 255.0  # Normalizar la imagen
img1_array = np.expand_dims(img1_array, axis=0) 

# Cargar y preprocesar la segunda imagen
img2 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
img2_array = image.img_to_array(img2) / 255.0  # Normalizar la imagen
img2_array = np.expand_dims(img2_array, axis=0)

# Unir las imágenes en un solo lote (batch)
img_batch = np.vstack([img1_array, img2_array])

# Hacer las predicciones
predictions = classifier.predict(img_batch)

# Mostrar las predicciones
print(predictions)
# Clasificacion binaria
predicted_class = (predictions > 0.5).astype('int')
