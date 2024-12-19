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

    # Define the blue color range
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([116, 255, 255])

    # Create a mask for blue color
    mask_blue = cv.inRange(frame_hsv, lower_blue, upper_blue)
    frame_blue = cv.bitwise_and(frame, frame, mask=mask_blue)

    # Display the frames
    cv.imshow('Original Frame', frame)
    cv.imshow('Blue Mask', mask_blue)
    cv.imshow('Blue Detection', frame_blue)

    # Exit the loop if 'ESC' is pressed
    if cv.waitKey(5) & 0xFF == 27:
        break

# Release the resources
cap.release()
cv.destroyAllWindows()
