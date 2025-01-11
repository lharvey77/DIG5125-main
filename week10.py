import cv2
import numpy as np
import time

# Initialize the video capture object
cap = cv2 . VideoCapture ( 'Videos / Cars . mp4')

# Read the first frame
ret , prev_frame = cap . read ()
if not ret :
    print ( " Failed to read the video " )
    cap . release ()
    exit ()
    
# Convert the first frame to grayscale
prev_frame = cv2 . cvtColor ( prev_frame , cv2 . COLOR_BGR2GRAY )

# Loop over all frames in the video
while True :
    ret , curr_frame = cap.read ()
    if not ret :
        break

# Convert to grayscale and apply Gaussian blur
    curr_frame_gray = cv2 . cvtColor ( curr_frame , cv2 . COLOR_BGR2GRAY )

# Frame differencing
    frame_diff = cv2 . absdiff ( prev_frame , curr_frame_gray )

# Display the result
    cv2 . imshow ( '' Frame Difference '' , frame_diff )

# Update previous frame
    prev_frame = curr_frame_gray

# Break the loop if ’q ’ is pressed
    if cv2 . waitKey (1) & 0xFF == ord ( 'q ') :
        break

# Release the video capture object and close all windows
cap.release ()
cv2.destroyAllWindows ()

