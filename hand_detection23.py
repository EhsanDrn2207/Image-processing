import cv2 as cv
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Initialize the video capture
cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")

# Initialize hand detector
detector = HandDetector(detectionCon=0.5, maxHands=2)

while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Detect hands in the frame
    hands, frame = detector.findHands(frame)
    
    if hands:
        # Process the first hand
        hand1 = hands[0]
        lmlist1 = hand1["lmList"]

        if len(lmlist1) > 8:  # Ensure the landmark list has enough points
            length, info, frame = detector.findDistance(lmlist1[4][:-1], lmlist1[8][:-1], frame)
            print(length)

        # Process the second hand if present
        if len(hands) > 1:
            hand2 = hands[1]
            lmlist2 = hand2["lmList"]
            if len(lmlist2) > 8:  # Ensure the landmark list has enough points
                length2, info2, frame = detector.findDistance(lmlist2[4][:-1], lmlist2[8][:-1], frame)
    
    # Display the frame
    cv.imshow('frame', frame)
    
    # Exit the loop if 'ESC' is pressed
    keyexit = cv.waitKey(5) & 0xFF
    if keyexit == 27:
        break

# Release the resources
cv.destroyAllWindows()
cap.release()
