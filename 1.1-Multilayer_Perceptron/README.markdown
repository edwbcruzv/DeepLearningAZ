# Clasificador de Frutas con Interfaz Gráfica

Esta aplicación utiliza un modelo de aprendizaje profundo (perceptrón multicapa) para clasificar imágenes de fresas y plátanos. La interfaz gráfica, desarrollada con `tkinter`, permite seleccionar hasta 20 imágenes (formatos `.jpg`, `.jpeg`, `.png`) y muestra las predicciones con sus probabilidades.

## Requisitos

- **Sistema operativo**: Windows, Linux o macOS.
- **Python**: Versión 3.12.0
- **Hardware**: CPU con soporte para instrucciones AVX (procesadores modernos post-2011).
- **Archivos necesarios**:
  - `main.py`: Script principal.
  - `fruit_classifier_model_improved.keras`: Modelo entrenado (debe estar en la misma carpeta que `main.py`).
  - Imágenes de prueba (`.jpg`, `.jpeg`, `.png`) para clasificar.

## Instalación

1. **Instalar Python**:
   - Descarga e instala Python 3.12.0 desde [python.org](https://www.python.org/downloads/).
   - Asegúrate de agregar Python al PATH durante la instalación.

2. **Crear un entorno virtual** (recomendado):
   ```bash
   python -m venv venv
   ```
   Activa el entorno:
   - Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```

3. **Instalar dependencias**:
   En el directorio del proyecto (donde está `requirements.txt`), ejecuta:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verificar archivos**:
   Asegúrate de que `main.py` y `fruit_classifier_model_improved.keras` estén en la misma carpeta.

## Ejecución

1. **Abrir la terminal**:
   Navega al directorio del proyecto:
   ```bash
   cd ruta/al/directorio
   ```

2. **Activar el entorno virtual** (si lo usaste):
   ```bash
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/macOS
   ```

3. **Ejecutar el script**:
   ```bash
   python main.py
   ```

4. **Usar la aplicación**:
   - Se abrirá una ventana con el título "Clasificador de Frutas".
   - Haz clic en "Seleccionar Imágenes (máx 20)" y elige imágenes (`.jpg`, `.jpeg`, `.png`).
   - La aplicación mostrará las imágenes en una cuadrícula con las predicciones ("fresa" o "plátano") y sus probabilidades.

## Solución de Problemas

- **Error: "No module named tensorflow"**:
  Asegúrate de que las dependencias se instalaron correctamente con `pip install -r requirements.txt`.
- **Error: "Failed to load the native TensorFlow runtime"**:
  Instala el redistribuible de Microsoft Visual C++ (Windows):
  ```bash
  pip install msvc-runtime
  ```
  O descárgalo desde: https://aka.ms/vs/17/release/vc_redist.x64.exe.
- **Error: Modelo no encontrado**:
  Verifica que `fruit_classifier_model_improved.keras` esté en la misma carpeta que `main.py`.
- **Imágenes no se cargan**:
  Asegúrate de que las imágenes sean válidas (`.jpg`, `.jpeg`, `.png`) y no estén corruptas.

## Notas

- La aplicación está limitada a 20 imágenes para evitar sobrecarga.
- El modelo clasifica imágenes como "fresa" o "plátano" basándose en un umbral de probabilidad de 0.5.
- Para mejores resultados, usa imágenes con fondo claro y frutas bien visibles, similares a las usadas en el entrenamiento.

## Contacto

Para dudas o problemas, contacta al desarrollador Edwin Cruz Villalba.