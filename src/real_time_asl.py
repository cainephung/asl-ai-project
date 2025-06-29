import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model
model = load_model("asl_model.h5")

# ASL label map (you can customize later)
labels = [chr(i) for i in range(65, 91)]  # A-Z

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame (adjust depending on your model input size)
    roi = frame[100:400, 100:400]  # Region of interest (square in center)
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1) / 255.0

    # Prediction
    predictions = model.predict(img)
    pred_class = np.argmax(predictions)
    pred_letter = labels[pred_class]

    # Draw prediction
    cv2.rectangle(frame, (100, 100), (400, 400), (255, 0, 0), 2)
    cv2.putText(frame, f"Predicted: {pred_letter}", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("ASL Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
