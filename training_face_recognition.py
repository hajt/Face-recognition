import cv2
import os
import numpy as np
import pickle

BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIRECTORY = os.path.join(BASE_DIRECTORY, 'training_data')

face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_index = 0
labels_index = {}
training_faces = []
faces_index = []

for root, dirs, files in os.walk(IMAGE_DIRECTORY):
    for file in files:
        path = os.path.join(root, file)
        label = os.path.basename(root)

        if label not in labels_index:
            labels_index[label] = current_index
            current_index += 1

        label_index = labels_index[label]
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image, 1.3, 5)
        for (x, y, w, h) in faces:
            face = cv2.resize(image[y:y + h, x:x + w], (250, 250))
            training_faces.append(face)
            faces_index.append(label_index)

with open("label.pickle", "wb") as f:
    pickle.dump(labels_index, f)

recognizer.train(training_faces, np.array(faces_index))
recognizer.save("trainer.yml")
