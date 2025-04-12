# import cv2
# import numpy as np
# import tensorflow as tf
# import pickle
# # from tensorflow.keras.models import load_model

# # Load pre-trained FaceNet model
# facenet_model = tf.keras.models.load_model('facenet_keras.h5')

# # Load trained KNN classifier
# with open('knn_classifier.pkl', 'rb') as f:
#     knn = pickle.load(f)

# # Load OpenCV's face detector
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Constants
# img_size = 160

# # Function to preprocess face images
# def preprocess_face(face_img):
#     face_img = cv2.resize(face_img, (img_size, img_size))
#     face_img = np.expand_dims(face_img, axis=0)
#     face_img = face_img.astype('float32') / 255.0
#     return face_img

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open video stream")
#     exit()

# print("[INFO] Starting real-time face recognition... Press 'q' to exit.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

#     for (x, y, w, h) in faces:
#         face_img = frame[y:y+h, x:x+w]

#         # Preprocess face
#         processed_face = preprocess_face(face_img)
#         embedding = facenet_model.predict(processed_face)[0]

#         # Predict using KNN
#         prediction = knn.predict([embedding])
#         label = prediction[0]

#         # Draw bounding box & label
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     cv2.imshow('Face Recognition', frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("[INFO] Real-time face recognition stopped.")


import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet

# Load FaceNet embedder
embedder = FaceNet()

# Load trained classifier
with open("face_recognition_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get face embeddings
    embeddings = embedder.embeddings([rgb_frame])

    if len(embeddings) > 0:
        pred_probs = model.predict_proba([embeddings[0]])[0]
        best_match_idx = np.argmax(pred_probs)
        confidence = pred_probs[best_match_idx]
        
        if confidence > 0.6:  # Confidence threshold
            predicted_name = label_encoder.inverse_transform([best_match_idx])[0]
        else:
            predicted_name = "Unknown"

        # Display name and confidence
        cv2.putText(frame, f"{predicted_name} ({confidence:.2f})", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
