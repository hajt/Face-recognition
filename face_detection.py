import cv2

image = cv2.imread('test_data/test1.jpg')
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

cv2.imshow('Face detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
