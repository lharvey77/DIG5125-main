import cv2
import numpy as np
import time

#Read frames and differences
'''cap = cv2.VideoCapture (0)
ret , prev_frame = cap . read ()
while cap . isOpened () :
    ret , frame = cap . read ()
    if not ret :
        break

    diff = cv2.absdiff ( prev_frame , frame )
    thresh = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ret,th1 = cv2.threshold(thresh,127,255,cv2.THRESH_BINARY)
    cv2 . imshow ('Frame Difference ', thresh )
    prev_frame = frame.copy ()
    if cv2.waitKey (30) & 0xFF == ord ('q'):
        break

cap . release ()
cv2 . destroyAllWindows ()'''


#Simple Tracking
'''kernel = np.ones((5, 5), np.uint8) 
def preprocess_frame ( frame ):
# Convert to grayscale and apply Gaussian blur
    gray = cv2 . cvtColor ( frame , cv2 . COLOR_BGR2GRAY )
    blurred = cv2 . GaussianBlur ( gray , (21 , 21) , 0)
    return blurred
# Initialize video capture
cap = cv2 . VideoCapture (0)
# Check if video opened successfully
if not cap . isOpened () :
    print (" Error opening video stream or file ")
    exit ()
# Get the FPS of the video
fps = cap . get ( cv2 . CAP_PROP_FPS )
# Read the first frame
ret , first_frame = cap . read ()
if not ret :
    print (" Failed to read the video ")
    cap . release ()
    exit ()
prev_frame = preprocess_frame ( first_frame )
while True :
    ret , frame = cap . read ()
    if not ret :
        break
    current_frame = preprocess_frame ( frame )
# Compute the difference between the current frame and the previous frame
    frame_diff = cv2 . absdiff ( prev_frame , current_frame )
# Apply thresholding to get a binary image
    thresh = cv2 . threshold ( frame_diff , 10 , 255 , cv2 . THRESH_BINARY )[1]
# Apply morphological operations to remove noise and fill in holes
# insert your code here
# Find contours on the thresholded image
    contours , _ = cv2 . findContours ( thresh . copy () , cv2 . RETR_EXTERNAL, cv2 . CHAIN_APPROX_SIMPLE )
    for contour in contours :
        if cv2 . contourArea ( contour ) < 800: # Minimum area threshold
            continue
        (x , y , w , h) = cv2 . boundingRect ( contour )
        cv2 . rectangle ( frame , (x , y ) , (x + w , y + h) , (0 , 255 , 0) ,2)
    vid_erosion  = cv2.erode(frame, kernel, iterations=1)
    vid_dilation = cv2.dilate(frame, kernel, iterations=1)
# Display the resulting frame
    cv2 . imshow (" Frame ", vid_erosion )
# cv2 . imshow (" Threshold ", thresh )
    key = cv2 . waitKey (int (1000/ fps ) ) & 0xFF
    if key == ord ('q'):
        break
    prev_frame = current_frame

cap . release ()
cv2 . destroyAllWindows ()'''


#Haar Cascade face tracking
# Load the pre - trained Haar Cascade model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2 . VideoCapture (0) # Use 0 for webcam . Replace with ’path_to_video . mp4 ' for a video file
while True :
    ret , frame = cap . read ()
    if not ret :
        break

# Convert to grayscale for face detection
    gray = cv2 . cvtColor ( frame , cv2 . COLOR_BGR2GRAY )

# Detect faces
    faces = face_cascade.detectMultiScale ( gray , 1.1 , 4)

# Draw rectangles around the faces
    for (x , y , w , h) in faces :
        face_region = frame[y:y+h, x:x+w]  #Extract the square from frame
        blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)  #Apply gaussian
        frame[y:y+h, x:x+w] = blurred_face  #Replace original region with blur

# Display the output
    cv2 . imshow ('Face Tracking ', frame )
    if cv2 . waitKey (1) & 0xFF == ord('q') :
        break
        
cap . release ()
cv2 . destroyAllWindows ()
