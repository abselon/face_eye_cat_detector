import cv2

eye_detector = cv2.CascadeClassifier('/content/haarcascade_eye.xml')
image = cv2.imread('/content/people1.jpg')
# image = cv2.resize(image, (800, 600))
image_mono = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_detector = cv2.CascadeClassifier('/content/haarcascade_frontalface_default.xml')
detections = face_detector.detectMultiScale(image_mono, scaleFactor = 1.3, minSize = (30,30)) #Default scaleFactor = 1.1

for (x, y, w, h) in detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,255), 2)

eye_detections = eye_detector.detectMultiScale(image_mono, scaleFactor = 1.1, minNeighbors = 10, maxSize = (70,70))

for (x, y, w, h) in eye_detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)


cv2.imshow(image)