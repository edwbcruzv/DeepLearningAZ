import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# CARGAR MODELO
model = load_model("fruit_classifier_model_improved.keras")
IMG_SIZE = (128, 128)

# CLASES
class_labels = ['fresa', 'plátano']  # 0: plátano, 1: fresa

# CARGAR Y MOSTRAR PREDICCIONES
test_dir = 'pruebas'
image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(
    ('.png', '.jpg', '.jpeg'))]

for fname in image_files:
    img_path = os.path.join(test_dir, fname)
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.image.per_image_standardization(img_array)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = class_labels[1] if prediction > 0.5 else class_labels[0]

    # --- MOSTRAR IMAGEN ---
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicción: {label} ({prediction:.2f})")
    plt.show()
