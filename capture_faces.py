# import cv2
# import os

# # Constants
# name = "person1"  # Change this for each person
# dataset_path = f'captured_images/{name}'
# os.makedirs(dataset_path, exist_ok=True)

# # Load OpenCV's pre-trained face detection model
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# count = 0
# print("[INFO] Capturing images... Press 'q' to stop.")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

#     for (x, y, w, h) in faces:
#         face_img = frame[y:y+h, x:x+w]
#         img_path = os.path.join(dataset_path, f"{count}.jpg")
#         cv2.imwrite(img_path, face_img)
#         count += 1

#         # Draw bounding box
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#     cv2.imshow('Face Capture', frame)

#     # Stop capturing when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# print(f"[INFO] Captured {count} images for {name}.")


import cv2
import numpy as np
import os
import pickle
from keras_facenet import FaceNet

# Initialize FaceNet model
embedder = FaceNet()

# Directory to save embeddings
output_dir = "face_embeddings"
os.makedirs(output_dir, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit()

name = input("Enter your name: ")  # Label for the face

face_data = []  # Store embeddings
count = 0

while count < 100:  # Capture 10 samples
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get embeddings
    embeddings = embedder.embeddings([rgb_frame])

    if len(embeddings) > 0:
        face_data.append((name, embeddings[0]))  # Store (label, embedding)
        count += 1
        print(f"✅ Captured {count}/100")

    cv2.imshow("Capturing Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save embeddings to file
with open(os.path.join(output_dir, f"{name}_embeddings.pkl"), "wb") as f:
    pickle.dump(face_data, f)

print(f"✅ Face embeddings for {name} saved successfully!")
