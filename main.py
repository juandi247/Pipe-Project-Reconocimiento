import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Suprimir mensajes informativos de TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Cargar el modelo entrenado
model = load_model("keras_Model.h5", compile=False)

# Cargar las etiquetas
with open("labels.txt", "r") as file:
    class_names = file.readlines()

# Colores para cada etiqueta (puedes cambiar los colores a tu preferencia)
colors = {
    0: (255, 0, 0),   # Cardboard - Rojo
    1: (0, 255, 0),   # Glass - Verde
    2: (0, 0, 255),   # Metal - Azul
    3: (255, 255, 0), # Paper - Amarillo
    4: (255, 0, 255), # Plastic - Magenta
    5: (0, 255, 255)  # Fruits - Cian
}

# Función para preprocesar las imágenes como en Teachable Machine
def preprocess_image(frame):
    size = (224, 224)  # Tamaño que acepta el modelo
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Convertir BGR (OpenCV) a RGB
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)      # Redimensionar y recortar
    image_array = np.asarray(image).astype(np.float32)               # Convertir a array de numpy
    normalized_image_array = (image_array / 127.5) - 1               # Normalizar imagen
    return normalized_image_array

# Abrir la cámara
cap = cv2.VideoCapture(0)

# Configurar la cámara para que funcione a una tasa de frames más alta (opcional)
cap.set(cv2.CAP_PROP_FPS, 30)  # Intenta establecer a 30 FPS

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocesar el frame para el modelo
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    preprocessed_frame = preprocess_image(frame)
    data[0] = preprocessed_frame

    # Hacer la predicción
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Quitar espacios o saltos de línea
    confidence_score = prediction[0][index]

    # Mostrar resultados en el frame solo si la confianza es mayor al 80%
    if confidence_score > 0.80:  
        color = colors.get(index, (255, 255, 255))  # Color según la clase
        label = f"{class_name}: {confidence_score:.2f}"
        
        # Dibujar el rectángulo alrededor de la imagen y el texto
        cv2.rectangle(frame, (50, 50), (400, 400), color, 2)
        cv2.putText(frame, label, (55, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Mostrar el frame con las predicciones
    cv2.imshow("Detección en tiempo real", frame)

    # Esperar un corto periodo para lograr una tasa de FPS más alta (aproximadamente cada ~33 ms para ~30 FPS)
    if cv2.waitKey(33) & 0xFF == ord('q'):  
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()