# --- IMPORTACIONES ---
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
import tensorflow as tf

# --- PARÁMETROS ---
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30

# GENERADORES
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    preprocessing_function=tf.image.per_image_standardization
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=tf.image.per_image_standardization
)

train_generator = train_datagen.flow_from_directory(
    'training',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    'test',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# PESOS DE CLASE
fresa_count = len(os.listdir('training/fresa'))
platano_count = len(os.listdir('training/platano'))
total = fresa_count + platano_count
class_weights = {
    0: (1 / platano_count) * (total / 2.0),
    1: (1 / fresa_count) * (total / 2.0)
} if fresa_count != platano_count else None

# MODELO MLP
model = Sequential([
    Flatten(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    Dense(512, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(256, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu', kernel_initializer='he_normal'),
    BatchNormalization(),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), 
             tf.keras.metrics.Recall()]
)

# CALLBACKS 
early_stop = EarlyStopping(patience=10, restore_best_weights=True, 
                           monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, 
                              min_lr=1e-6)

# ENTRENAMIENTO
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weights
)

# GUARDAR MODELO
model.save("fruit_classifier_model_improved.keras")
print("✅ Modelo entrenado y guardado como 'fruit_classifier_model_improved.keras'")

plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()
