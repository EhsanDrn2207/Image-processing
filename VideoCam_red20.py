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

    # Define the red color range
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create a mask for red color
    mask_red1 = cv.inRange(frame_hsv, lower_red1, upper_red1)
    mask_red2 = cv.inRange(frame_hsv, lower_red2, upper_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    frame_red = cv.bitwise_and(frame, frame, mask=mask_red)

    # Display the frames
    cv.imshow('Original Frame', frame)
    cv.imshow('Red Mask', mask_red)
    cv.imshow('Red Detection', frame_red)

    # Exit the loop if 'ESC' is pressed
    if cv.waitKey(5) & 0xFF == 27:
        break

# Release the resources
cap.release()
cv.destroyAllWindows()
