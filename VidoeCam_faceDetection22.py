import cv2 as cv
import numpy as np

# Load Haar Cascade classifiers for face, eye, and smile detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')

# Initialize the video capture
cap = cv.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise Exception("Could not open video device")

while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert the frame to grayscale
    frame_gr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(frame_gr, scaleFactor=1.3, minNeighbors=5)
    
    # Iterate over the detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face and add label
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv.putText(frame, 'Face', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
        
        # Region of Interest (ROI) for grayscale and color frames
        frame_gr_roi = frame_gr[y:y + h, x:x + w]
        frame_roi = frame[y:y + h, x:x + w]
        
        # Detect eyes within the face ROI
        eyes = eye_cascade.detectMultiScale(frame_gr_roi, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around each detected eye and add label
            cv.rectangle(frame_roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            cv.putText(frame_roi, 'Eye', (ex, ey - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Detect smiles within the face ROI with adjusted parameters
        smiles = smile_cascade.detectMultiScale(frame_gr_roi, scaleFactor=1.9, minNeighbors=22, minSize=(40, 40))
        for (sx, sy, sw, sh) in smiles:
            # Draw rectangle around each detected smile and add label
            cv.rectangle(frame_roi, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
            cv.putText(frame_roi, 'Smile', (sx, sy - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Display the frame with detections
    cv.imshow('Face, Eye, and Smile Detection', frame)
    
    # Exit the loop if 'ESC' is pressed
    if cv.waitKey(5) & 0xFF == 27:
        break

# Release the resources
cap.release()
cv.destroyAllWindows()
