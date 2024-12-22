import cv2 as cv
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.FaceMeshModule import FaceMeshDetector

# Initialize the video capture
cap = cv.VideoCapture(0)  # Use 0 for primary webcam
if not cap.isOpened():
    raise Exception("Could not open video device")

# Initialize detectors
hand_detector = HandDetector(detectionCon=0.5, maxHands=2)
face_detector = FaceDetector()
mesh_detector = FaceMeshDetector(maxFaces=1)

while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect hands in the frame
    hands, frame = hand_detector.findHands(frame)

    # Detect faces
    frame, bboxs = face_detector.findFaces(frame)

    # Detect face mesh
    frame, faces = mesh_detector.findFaceMesh(frame)

    # Process detected hands
    if hands:
        # Process the first hand
        hand1 = hands[0]
        lmlist1 = hand1["lmList"]

        if len(lmlist1) > 8:  # Ensure the landmark list has enough points
            length, info, frame = hand_detector.findDistance(lmlist1[4][:-1], lmlist1[8][:-1], frame)
            # print(f"Distance between thumb and index finger of Hand 1: {length:.2f} pixels")


        # Process the second hand if present
        if len(hands) > 1:
            hand2 = hands[1]
            lmlist2 = hand2["lmList"]
            if len(lmlist2) > 8:  # Ensure the landmark list has enough points
                length2, info2, frame = hand_detector.findDistance(lmlist2[4][:-1], lmlist2[8][:-1], frame)
                # print(f"Distance between thumb and index finger of Hand 2: {length2:.2f} pixels")

    # Process detected faces
    if bboxs:
        for bbox in bboxs:
            center = bbox["center"]
            cv.circle(frame, center, 5, (255, 0, 255), cv.FILLED)
            cv.putText(frame, "Face", (bbox["bbox"][0], bbox["bbox"][1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9,
                       (255, 0, 255), 2)

    # Display the frame
    cv.imshow('Hand, Face, and Face Mesh Detection', frame)

    # Exit the loop if 'ESC' is pressed
    keyexit = cv.waitKey(5) & 0xFF
    if keyexit == 27:
        break

# Release the resources
cv.destroyAllWindows()
cap.release()
