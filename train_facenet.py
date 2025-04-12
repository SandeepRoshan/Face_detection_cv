# import os
# import cv2
# import numpy as np
# import tensorflow as tf
# # from tensorflow.keras.models import load_model
# from sklearn.neighbors import KNeighborsClassifier
# import pickle

# import os
# print(tf.__version__) 
# file_size = os.path.getsize("facenet_keras.h5") / (1024 * 1024)  # Convert to MB
# print(f"File size: {file_size:.2f} MB")

# # Load FaceNet model
# facenet_model = tf.keras.models.load_model('facenet_keras.h5')  # Ensure you have the FaceNet model

# # Constants
# img_size = 160
# dataset_path = './captured_images'
# embedding_dict = {}
# labels = []

# # Function to preprocess images
# def preprocess_face(face_img):
#     face_img = cv2.resize(face_img, (img_size, img_size))
#     face_img = np.expand_dims(face_img, axis=0)
#     face_img = face_img.astype('float32') / 255.0
#     return face_img

# # Extract face embeddings
# for person in os.listdir(dataset_path):
#     person_path = os.path.join(dataset_path, person)
    
#     for img_name in os.listdir(person_path):
#         img_path = os.path.join(person_path, img_name)
#         img = cv2.imread(img_path)

#         # Extract embedding
#         face = preprocess_face(img)
#         embedding = facenet_model.predict(face)[0]

#         embedding_dict[img_path] = embedding
#         labels.append(person)

# # Convert embeddings to numpy array
# X = np.array(list(embedding_dict.values()))
# y = np.array(labels)

# # Train KNN classifier
# knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
# knn.fit(X, y)

# # Save trained classifier
# with open('knn_classifier.pkl', 'wb') as f:
#     pickle.dump(knn, f)

# print("[INFO] Model trained and saved as knn_classifier.pkl")


import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load embeddings
face_embeddings = []
labels = []

for file in os.listdir("face_embeddings"):
    if file.endswith(".pkl"):
        with open(os.path.join("face_embeddings", file), "rb") as f:
            data = pickle.load(f)
            for label, embedding in data:
                face_embeddings.append(embedding)
                labels.append(label)

# Convert to numpy arrays
X = np.array(face_embeddings)
y = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM classifier
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model & label encoder
with open("face_recognition_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model and label encoder saved successfully!")
