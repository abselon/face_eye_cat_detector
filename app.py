from flask import Flask, render_template, request, url_for
import cv2
import numpy as np
import os

app = Flask(__name__)

# Ensure there's a 'static' folder in the same directory as this script
app.config['UPLOAD_FOLDER'] = 'static/images'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        # Assuming the image is sent as a file in the POST request
        file = request.files['image']
        # Read the image file
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        
        # Your existing eye detection code
        eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')
        image_mono = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cat_detector = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
        detections = face_detector.detectMultiScale(image_mono, scaleFactor=1.3, minSize=(30,30))

        for (x, y, w, h) in detections:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,255), 2)

        eye_detections = eye_detector.detectMultiScale(image_mono, scaleFactor=1.1, minNeighbors=10, maxSize=(70,70))

        for (x, y, w, h) in eye_detections:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)

        cat_detections = cat_detector.detectMultiScale(image_mono, scaleFactor=1.3)

        for (x, y, w, h) in cat_detections:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,0), 2)

         # Save the result image to the static folder if both face or eyes  detected
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected.jpg')
        cv2.imwrite(output_path, image)
        
        return render_template('index.html', filename='detected.jpg')


        

if __name__ == '__main__':
    app.run(debug=True)
