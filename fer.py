import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model_path = 'C:/Users/ankus/PycharmProjects/pythonProject3/FER_Final.h5'
model = load_model(model_path)


# Define the labels for your seven classes
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale (if needed) and preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Assume the model expects 48x48 pixel images as input (adjust as needed)
    face = cv2.resize(gray, (48, 48))
    face = face.astype('float32') / 255  # Normalize
    face = np.expand_dims(face, axis=0)  # Add batch dimension
    face = np.expand_dims(face, axis=-1)  # Add channel dimension

    # Make a prediction
    prediction = model.predict(face)
    class_index = np.argmax(prediction, axis=1)
    predicted_class = class_labels[class_index[0]]

    # Display the resulting frame with the predicted emotion
    cv2.putText(frame, predicted_class, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Facial Emotion Recognition', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
