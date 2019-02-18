import cv2
import pickle

with open("label.pickle", "rb") as f:
    imported_labels = pickle.load(f)
    labels_index = {v: k for k, v in imported_labels.items()}

face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

image = cv2.imread('test_data/test3.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
for (x, y, w, h) in faces:
    roi = image_gray[y:y+h, w:x+w]
    index, confidence = recognizer.predict(roi)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

    if confidence < 100:
        cv2.putText(image, labels_index[index], (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, (str(round(confidence)) + "%"), (x, y+h),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

cv2.imshow('Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
