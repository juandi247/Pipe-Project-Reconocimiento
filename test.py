import os
import cv2
import numpy as np
import serial
import time
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

# Inicializar la conexión serie con Arduino (ajusta 'COM3' al puerto correcto)
arduino = serial.Serial('COM6', 9600) 
time.sleep(2) # Esperar a que se establezca la conexión

# Función para preprocesar las imágenes como en Teachable Machine
def preprocess_image(frame):
    size = (224, 224)  
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1  
    return normalized_image_array

# Abrir la cámara
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    preprocessed_frame = preprocess_image(frame)
    data[0] = preprocessed_frame

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Mostrar resultados en el frame solo si la confianza es mayor al 80%
    if confidence_score > 0.90:
        color = colors.get(index, (255, 255, 255))
        label = f"{class_name}: {confidence_score:.2f}"
        
        cv2.rectangle(frame, (50, 50), (400, 400), color, 2)
        cv2.putText(frame, label, (55, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Enviar comando al Arduino según la clase detectada
        if index in [4, 3]: # Plastic o Paper -> LED1
            arduino.write(b'1')
        elif index in [2, 1]: # Metal o Glass -> LED2
            arduino.write(b'2')
        elif index == 5: # Fruits -> LED3
            arduino.write(b'3')
    else: 
        arduino.write(b'0') # Apagar todos los LEDs si no coincide con ninguna clase

    cv2.imshow("Detección en tiempo real", frame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
arduino.close() # Cerrar la conexión con Arduino al final del script.