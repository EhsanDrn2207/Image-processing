import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt 

def QR_decode(image): 
    # Find barcodes and QR codes
    decoded_objects = pyzbar.decode(image)
    # Print results
    for obj in decoded_objects:
        print('Type : ', obj.type)
        print('Data : ', obj.data.decode('utf-8'), '\n')
    return decoded_objects

def display(image, decoded_objects):
    # Loop over all decoded objects
    for decoded_object in decoded_objects:
        points = decoded_object.polygon
        # If the points do not form a quad, find convex hull
        if len(points) > 4: 
            hull = cv.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else: 
            hull = points
        # Number of points in the convex hull
        n = len(hull)
        # Draw the convex hull
        for j in range(n):
            cv.line(image, hull[j], hull[(j+1) % n], (255, 0, 0), 3)
    return image

# Initialize video capture
cap = cv.VideoCapture(0)  # Change to 0 if 1 does not work
if not cap.isOpened():
    raise Exception("Could not open video device")

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert frame to grayscale
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Decode QR codes in the frame
    decoded_objects = QR_decode(frame_gray)
    
    # Display decoded QR codes and draw bounding boxes
    for decoded_object in decoded_objects:
        points = decoded_object.polygon
        # Find convex hull if necessary
        if len(points) > 4:
            hull = cv.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points
        
        # Draw convex hull
        for j in range(len(hull)):
            cv.line(frame, hull[j], hull[(j+1) % len(hull)], (255, 0, 0), 3)
        
        x, y = decoded_object.rect.left, decoded_object.rect.top
        print('Type : ', decoded_object.type)
        print('Data : ', decoded_object.data.decode('utf-8'), '\n')
        
        barcode_data = decoded_object.data.decode('utf-8')
        cv.putText(frame, barcode_data, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv.LINE_AA)
    
    # Display the frame
    cv.imshow('QR Code Detection', frame)
    
    # Exit the loop if 'ESC' is pressed
    if cv.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv.destroyAllWindows()
