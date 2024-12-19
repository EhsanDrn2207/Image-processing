import cv2 as cv
import numpy as np 

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
    
    # Convert the frame to HSV color space
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Define the green color range
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    # Create a mask for green color
    mask_green = cv.inRange(frame_hsv, lower_green, upper_green)
    frame_green = cv.bitwise_and(frame, frame, mask=mask_green)

    # Display the frames
    cv.imshow('Original Frame', frame)
    cv.imshow('Green Mask', mask_green)
    cv.imshow('Green Detection', frame_green)

    # Exit the loop if 'ESC' is pressed
    if cv.waitKey(5) & 0xFF == 27:
        break

# Release the resources
cap.release()
cv.destroyAllWindows()
