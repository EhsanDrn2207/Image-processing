import cv2 as cv
import numpy as np
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

# Initialize the video capture
cap = cv.VideoCapture(0)  # Use 0 for primary webcam
if not cap.isOpened():
    raise Exception("Could not open video device")

# Initialize detectors
face_detector = FaceDetector()
mesh_detector = FaceMeshDetector(maxFaces=1)

while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Detect faces
    frame, bboxs = face_detector.findFaces(frame)
    
    # Detect face mesh
    frame, faces = mesh_detector.findFaceMesh(frame)

    # Process detected faces
    if bboxs:
        for bbox in bboxs:
            center = bbox["center"]
            cv.circle(frame, center, 5, (255, 0, 255), cv.FILLED)
            cv.putText(frame, "Face", (bbox["bbox"][0], bbox["bbox"][1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    
    # Display the frame
    cv.imshow('Face and Face Mesh Detection', frame)
    
    # Exit the loop if 'ESC' is pressed
    keyexit = cv.waitKey(5) & 0xFF
    if keyexit == 27:
        break

# Release the resources
cv.destroyAllWindows()
cap.release()
