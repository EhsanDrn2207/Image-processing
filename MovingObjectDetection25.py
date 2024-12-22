import numpy as np
import cv2 as cv


def rescale_frame(frame, percent=20):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)


# Initialize video capture
cap = cv.VideoCapture("street_camera2.mp4")
fps = cap.get(cv.CAP_PROP_FPS)
print(fps)

while True:
    # Read two consecutive frames
    rec1, frame1 = cap.read()
    rec2, frame2 = cap.read()

    if not rec1 or not rec2:
        print("Failed to grab frame")
        break

    # Rescale frames
    resized_frame1 = rescale_frame(frame1)
    resized_frame2 = rescale_frame(frame2)

    # Compute frame difference
    frame_diff = cv.absdiff(resized_frame1, resized_frame2)
    frame_diff_gr = cv.cvtColor(frame_diff, cv.COLOR_BGR2GRAY)
    blurred_frame = cv.GaussianBlur(frame_diff_gr, (9, 9), 1)
    _, mask = cv.threshold(blurred_frame, 10, 255, cv.THRESH_BINARY)

    # Find contours
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected contours
    for contour in contours:
        if cv.contourArea(contour) > 1000:
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(resized_frame1, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display frames
    cv.imshow('Original Frame with Detections', resized_frame1)
    cv.imshow('Frame Difference', frame_diff)
    cv.imshow('Binary Mask', mask)

    # Exit the loop if 'ESC' is pressed
    keyexit = cv.waitKey(5) & 0xFF
    if keyexit == 27:
        break

# Release resources
cv.destroyAllWindows()
cap.release()
