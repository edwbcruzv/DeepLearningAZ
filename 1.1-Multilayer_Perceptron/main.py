import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Cargar modelo
try:
    model_path = os.path.join(os.path.dirname(__file__), "fruit_classifier_model_improved.keras")
    model = load_model(model_path)
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    messagebox.showerror("Error", f"No se pudo cargar el modelo: {e}")
    exit(1)

IMG_SIZE = (128, 128)
class_labels = ['fresa', 'plátano']  # 0: plátano, 1: fresa

# Función para predecir etiqueta
def predict_label(img_path):
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = tf.image.per_image_standardization(img_array)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array, verbose=0)[0][0]
        label = class_labels[1] if prediction > 0.5 else class_labels[0]
        return label, prediction
    except Exception as e:
        return "Error", f"No se pudo procesar la imagen: {e}"

# Clase para scrollable frame
class ScrollableFrame(tk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self, height=500)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

# Función para seleccionar imágenes y mostrarlas
def select_images():
    file_paths = filedialog.askopenfilenames(
        title="Seleccionar imágenes",
        filetypes=[("Imágenes", "*.jpg *.jpeg *.png")]
    )
    if not file_paths:
        return
    
    file_paths = file_paths[:20]  # Limitar a 20 imágenes

    # Limpiar contenido previo
    for widget in frame.scrollable_frame.winfo_children():
        widget.destroy()

    # Mostrar imágenes y etiquetas en grid 4 columnas
    max_columns = 4
    thumb_size = 150
    photos.clear()  # Evitar que imágenes se recolecten con garbage collector
    
    for idx, img_path in enumerate(file_paths):
        label_pred, pred_prob = predict_label(img_path)

        # Cargar imagen y redimensionar thumbnail
        try:
            img_pil = Image.open(img_path)
            img_pil.thumbnail((thumb_size, thumb_size))
            img_tk = ImageTk.PhotoImage(img_pil)
            photos.append(img_tk)  # Guardar referencia
        except Exception as e:
            label_pred = "Error"
            pred_prob = f"No se pudo cargar la imagen: {e}"

        # Crear frame para cada imagen + etiqueta
        container = tk.Frame(frame.scrollable_frame, bd=2, relief="groove", padx=5, pady=5)
        container.grid(row=idx // max_columns, column=idx % max_columns, padx=5, pady=5)

        if label_pred != "Error":
            # Label imagen
            img_label = tk.Label(container, image=img_tk)
            img_label.pack()
            # Label texto con la predicción
            text_label = tk.Label(container, text=f"{label_pred}\n({pred_prob:.2f})", font=("Helvetica", 12))
        else:
            # Mostrar error
            text_label = tk.Label(container, text=pred_prob, font=("Helvetica", 12), fg="red")
        text_label.pack()

# Interfaz principal
root = tk.Tk()
root.title("Clasificador de Frutas")
root.geometry("650x600")

# Título
title_label = tk.Label(root, text="Clasificador de frutas con IA", font=("Helvetica", 18))
title_label.pack(pady=10)

# Botón seleccionar imágenes
btn = tk.Button(root, text="Seleccionar Imágenes (máx 20)", command=select_images, font=("Helvetica", 14), bg="lightgreen")
btn.pack(pady=10)

# Frame scrollable para imágenes
frame = ScrollableFrame(root)
frame.pack(fill="both", expand=True, padx=10, pady=10)

photos = []  # Para guardar referencias a imágenes y que no las borre el GC

root.mainloop()