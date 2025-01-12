import cv2 #importing OpenCV for video altering
import numpy as np #importing Numpy for maths for kernels and centroids
import matplotlib.pyplot as plt #importing matplotlib for plotting window to display video
from matplotlib.backend_bases import MouseEvent #importing MouseEvent for additional functionality
from matplotlib.widgets import Button #importing ButtonEvent for additional functionality

'''Video with object tracking
User can select filter for entire screen
User can select filter for tracked object
User can pick with object to visualise tracking for
User can pick video file to upload'''

'''Declaring variables'''
kernel = np.ones((5, 5), np.uint8) #a kernel is a matrix used to apply filters to images
#np.ones will fill the matrix with 1's. 5,5 means the matrix is 5 by 5 large. np.uint8 is the data type

blur_enabled = False #variable for enabling or disabling Gaussian blur filter
edge_enabled = False #variable for enabling or disabling Edge Detection

maximum_points = 50 #to prevent the screen from filling with too many circles, the tracking path will delete the oldest circle past 50
track_paths = [] #an array for the paths to visualise tracking
selected_obj = None #a variable for the objects being tracked. 
current_centroids = [] #an array for the current centroids (middle point of objects) on screen

'''Declaring functions'''
def preproc_frame(frame): #a function to prepare every frame for object tracking and effects (pre process)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #turns the frame black and white
    blur = cv2.GaussianBlur(gray, (21, 21), 0) #gaussian blurs the frame
    return blur #returns the b/w blurred frame

def get_centroid(contour): #function for getting centroid (contour = outline (points of shape), centroid = center point) 
    moments = cv2.moments(contour) #moments to find area of object
    if moments["m00"] != 0:  #'m00' is for finding the area; as long as the area is not 0, continue with code
        cx = int(moments["m10"] / moments["m00"]) #'m10' is for finding the 'center of gravity of the outline'; this finds the x coordinate of the centroid
        cy = int(moments["m01"] / moments["m00"]) #'m01' is for finding the 'center of gravity of the outline'; this finds the y coordinate of the centroid
        return cx, cy #return points 'centroid x axis' and #centroid y axis'
    return None #otherwise, return nothing

def visual_tracking_path(frame, object_index): #function for visualising the objects tracking path
    if object_index is not None: #as long as there is an object to track,
        for path in track_paths[object_index]: #for every movement/path in the track_paths array,
            cv2.circle(frame, path, 3, (0, 0, 255), -1) #draw a circle in every movement/path location
            
def apply_blur(event): #function for applying the gaussian blur
    global blur_enabled #use the blur_enabled as a global variable so it can be used/seen outside the function (this will trigger the if statement in the main loop)
    blur_enabled = not blur_enabled  #to toggle the blur state
    print("Gaussian Blur", "Enabled" if blur_enabled else "Disabled") #print a message in the terminal whether the blur is enabled or not 
    
def apply_edgedet(event): #function for applying edge detection
    global edge_enabled #use the edge_enabled as a global variable so it can be used/seen outside the function (this will trigger the if statement in the main loop)
    edge_enabled = not edge_enabled #to toggle the edge detection state
    print("Edge Detection", "Enabled" if edge_enabled else "Disabled") #print a message in the terminal whether the edge detection is enabled or not 
    
def on_click(event: MouseEvent): #a function for clicking on the matplotlib display
    global selected_obj #use the selected_obj as a global variable so it can be used/seen outside the function (this will trigger the if statement in the main loop)
    if event.inaxes != ax: #check if the mouse click happened inside the video (inside the axes 'instance')
        return #if the event (mouse clicking) did not happen inside the axes, return nothing and close
    
    click_x, click_y = int(event.xdata), int(event.ydata) #variables for the x and y coordinates for the mouse clicking
    min_distance = float('inf') #variable. the minimum distance is infinity because the program doesn't know what the minimum distance could be. this allows the minimum distance between the
                                #centroid (from the array, current_centroids) and the users click to be anything (within infinity...)
    closest_obj_index = None #variable initialised as None (empty). it holds the index (place in list of current_centroids) of the object (centroid) closest to the click. that object is the one
                            #that will have its track visualised
    
    for i, centroid in enumerate(current_centroids): #loop (iterate (enumerate)) through the current_centroids array. this array contains the x ([0]) and y([1]) coordinates of the centroid
        distance = np.sqrt((centroid[0] - click_x)**2 + (centroid[1] - click_y)**2) #calculte the distance (with a euclidean formula) between the mouse click and the centroid 
        #square root of (centroid x co-ord - click x co-ord)squared + (centroid y co-ord - click y co-ord)squared
        if distance < min_distance: #if the distance is within minimum_distance (starting as infinity but changing during the loop)
            min_distance = distance #the minimum_distance becomes the current distance
            closest_obj_index = i #closest_obj_index is updated to the current index (i). this object (centroid) is now the closest to the mouse click so it will be selected to be tracked
            #eg, the second centroid in the array current_centroids is the one clicked on. i becomes 1, which is the second item in the array

    if closest_obj_index is not None: #if the closest_obj_index exists and is not nothing/None, an object was found with a centroid close enough to the mouse click
        selected_obj = closest_obj_index #so the selected_obj (the object that is now selected) becomes the array pointer value in closest_obj_index
        print(f"Selected object {selected_obj + 1}") #the terminal prints which object was selected and adds 1 because the first objects index is 0

'''Setup'''
filepath = input("Enter the path of your file. Include the filetype (EG .mp4). Enter '0' if you are using a webcam.") #allows the user to input their own filepath for the video
cap = cv2.VideoCapture(filepath) #assigns the value given by the user (filepath) to the video capture
if not cap.isOpened(): #if the cap can't be opened,
    print("Error reading video camera or file.") #return error message
    exit() #exit program

fps = cap.get(cv2.CAP_PROP_FPS) #use cv2 to retrieve the frames per second
ret, first_frame = cap.read() #"ret" returns if the frame is available (boolean), "first_frame" is returning the first/current frame = the video cap is read
prev_frame = preproc_frame(first_frame) #the frame is updated to be pr-processed

fig, ax= plt.subplots(figsize=(12, 8)) #Creating display with matplotlib, with a size declared so it's bigger than the smaller default size
    
bl_button = plt.axes([0.7, 0.05, 0.2, 0.075]) #places a button: 0.7 is the x position, 0.05 is the y position, 0.2 is the width, 0.075 is the height
blurbutton = Button(bl_button, "Toggle Gaussian Blur") #the text on the button: this is the button for toggling Gaussian Blur
blurbutton.on_clicked(apply_blur) #when the button is clicked, it triggers the function for the apply_blur function

edg_button = plt.axes([0.3, 0.05, 0.2, 0.075]) #places a button: 0.3 is the x position, 0.05 is the y position, 0.2 is the width, 0.075 is the height
edgebutton = Button(edg_button, "Toggle Edge Detection") #the text on the button: this is the button for toggling Edge Detection
edgebutton.on_clicked(apply_edgedet) #when the button is clicked, it triggers the function for the apply_edgedet function

fig.canvas.mpl_connect("button_press_event", on_click) #this makes a connection between a click (button_press_event) and the function for what will happen when the mouse is clicked (on_click).
                                                        #so when the mouse is clicked, the event connection (fig.canvas.mpl_connect) connects button_press_event and on_click

'''Main loop'''
while True: #this loops forever. we need this to read the frames from the video until it ends
    ret, frame = cap.read() #ret is a boolean for if the frame has been returned, frame is being returned, and this is what the cap (capture) is reading
    if not ret: #if ret is not true, and there isn't a frame to read
        print("Error reading video camera or file.") #print error in terminal
        break #finish if statement

    fig.canvas.manager.set_window_title("Video - Object Tracking") #changes the name of the window title
    current_frame = preproc_frame(frame) #the current frame being displayed is constantly changing in the loop
                                        #so the current_frame is loading the pre_processed frame (which is loading the frame) frame which will be used for the filters

    frame_diff = cv2.absdiff(prev_frame, current_frame) #this finds the differences between the current frame and the last 
    thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)[1] #thresholding the differences between the images and storing it in the thresh variable.
                                    #10 is the threshold, so pixel values below 10 will be changed to black.
                                    #255 is the maximum value, so pixel values above 10 will be changed to white (255)

    thresh = cv2.erode(thresh, kernel, iterations=1) #erosion will remove small white noise of the thresholded image and uses the kernel declared at the start to do so.
    thresh = cv2.dilate(thresh, kernel, iterations=1) #erosion has shrinked the image (frame), so dilation will increase the size again, using the kernel declared at the start


    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #this is finding the contours in the frame from a copy of the thresholded image
                                    #RETR_EXTERNAL is the contour-retrieval-type and will only retrieve the 'extreme outer contours',
                                    #CHAIN_APPROX_SIMPLE is the contour-approximation-method and reduces the points of the contour into only end points (EG 'a rectangle is left with 4 points')
    current_centroids.clear() #everytime a centroid/object disappears from view, this will clear that centroid from the array and the counter at the bottom of the image

    for contour in contours: #while looping through every contour, 
        if cv2.contourArea(contour) < 2300: #if the area of the contour is less than 2300 pixels,
            continue #carry on without doing anything
        (x, y, w, h) = cv2.boundingRect(contour) #otherwise, find the bounding rectangle of the contour. the x and y are for the top left of the rectangle and w + h are the width and height

        centroid = get_centroid(contour) #get the centroid using the 'get_centroid' fuction with the contour
        if centroid: #if there is a centroid,
            if len(current_centroids) < 25: #and if the amount of current_centroids (visible centroids) is less than 9, (to reduce clutter)
                current_centroids.append(centroid) #append (add) the newest centroid to the array of current_centroids (centroids that are visible on the screen)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) #build the rectangle that will be seen on the screen. place it on the frame at x, y co-ordinates
                                #add the x co-ord and the width for the starting point. add the y co-ord and the height for the ending point. 0, 255, 0 is the colour (red). 2 is the thickness
                object_number = len(current_centroids) #len to count the amount of objects in the current_centroids array. this is put in the object_number variable which is used to display the number of the object
                cv2.putText(frame, str(object_number), (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                #^ displays text inside the bounding box (contours, using the centroid). frame = what the text is put on top off. str(object_nuber) = the text being written (the number of the object)
                #centroid[0] - 10, centroid[1] - 10) = using the centroid to determine the co-ord of the bottom left (starting point) point of the text
                #cv2.FONT_HERSHEY_SIMPLEX = font, 1 = fontscale (size of text), (0, 0, 255) = colour, 2 = thickness

    while len(track_paths) < len(current_centroids): #as long as the amount of current centroids on the screen are less than the amount of visualised tracking paths,
        track_paths.append([]) #add an empty place for the new tracking path VVV

    for i, obj in enumerate(current_centroids): #for every object (and i = current placement in track_paths) while enumerating the current_centroids array,
        if i < 25: #and if the amount of objects is less than 25
            track_paths[i].append(obj) #append the object (obj) in the current tracking path index (track_paths[i])
            if len(track_paths[i]) > maximum_points: #if the amount of tracking paths (the dots that are visualising) go over the maximum_points (50)
                track_paths[i].pop(0) #pop (delete) the first added track_path 

    if selected_obj is not None: #as long as there is a selected object to track,
        visual_tracking_path(frame, selected_obj) #perform the visual_tracking_path function using frame and selected_obj (the object that has been clicked on)

    if blur_enabled: #if the button to enable blur has been clicked,
        for contour in contours: #and for every contour in the contours array,
            if cv2.contourArea(contour) < 2300: #and if the area of the contour is under 2300 pixels,
                continue #do nothing
            (x, y, w, h) = cv2.boundingRect(contour) #otherwise, use the x, y, width, and height to create a bounding rectangle
            blurred_region = cv2.GaussianBlur(frame[y:y+h, x:x+w], (99, 99), 30) #created a variable for the area that will be blurred. use cv2.GaussianBlur on the frame
                                    #and declare that for the y direction, start at the y co-ord and finish at y + height; declare for the x direction, start at the x co-ord and finish at x + width
                                    #(99, 99) is the kernel, which must be positive and odd
                                    #30 is the border type
            frame[y:y+h, x:x+w] = blurred_region #the blurred region becomes the y:y+h, x:x+w area of the frame
            
    if edge_enabled: #if the button to enable edge detection has been clicked,
        for contour in contours: #and for every contour in the contours array,
            if cv2.contourArea(contour) < 2300: #and if the area of the contour is under 2300 pixels,
                continue #do nothing
            (x, y, w, h) = cv2.boundingRect(contour) ##otherwise, use the x, y, width, and height to create a bounding rectangle
            edge_region = cv2.Canny(frame, 10, 70) #create a variable using canny edge detection on the frame. 10 = minimum value (this will be black), 70 = maximum value (this will be white)
            ret, frame = cv2.threshold(edge_region, 70, 255, cv2.THRESH_BINARY) #as long as ret is true, and using the frame, threshold the edge_region variable. anything under 70 will be black and anything above will be white (255)

    object_count = len(current_centroids) #make a variable for counting the amount of objects on screen. do this by using len to count the amount in current_centroids
    cv2.putText(frame, f"Objects counted: {object_count}", (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #^ displays text to tell the user how many objects have been counted. object_count = the amount of items in the current_centroids array
    #10, 1000 = co-ord of the bottom left (starting point) point of the text. 10 = x, 1000 = y
    #cv2.FONT_HERSHEY_SIMPLEX = font, 1 = fontscale (size of text), (255, 255, 255) = colour, 2 = thickness
    cv2.putText(frame, "Click on object to visualise tracking.", (10, 1030), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) #
    #^ displays text to tell the user how to select an object to track
    #10, 1030 = co-ord of the bottom left (starting point) point of the text. 10 = x, 1000 = y
    #cv2.FONT_HERSHEY_SIMPLEX = font, 1 = fontscale (size of text), (255, 255, 255) = colour, 2 = thickness
    
    ax.clear() #clear the screen of the last frame
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #display the next frame, in RGB colour
    ax.set_title(f"Tracking {selected_obj + 1 if selected_obj is not None else 'nothing'}") #set the title of the axes as Tracking the selected_obj + 1, because the array index starts at 0. If there is nothing to track, say 'nothing'
    plt.pause(1 / fps) #display 1 frame per seciond by pausing every second

    key = cv2.waitKey(int(1000 / fps)) & 0xFF #wait for the q key to be pressed at any moment during the fps (like duration) of the video
    if key == ord('q'): #if the q key is pressed
        break #stop the video

    prev_frame = current_frame #at the end of every iteration of the while loop, the frame updates

cap.release() #release the video capture (end)
cv2.destroyAllWindows() #close the windows at the end

''' Websites used
https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
https://byjus.com/maths/centroid/
https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
https://en.codingwalks.com/entry/OpenCV-Python-Contour-Detection-and-Labeling-1
https://matplotlib.org/stable/users/explain/figure/event_handling.html
https://byjus.com/maths/euclidean-distance/
https://matplotlib.org/stable/users/explain/figure/event_handling.html
https://www.futurelearn.com/info/courses/introduction-to-image-analysis-for-plant-phenotyping/0/steps/305359
https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
https://www.geeksforgeeks.org/image-thresholding-in-python-opencv/
https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/
https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
https://medium.com/@akash555bhiwgade/edge-detection-with-15-lines-of-python-code-using-opencv-and-webcam-8f980c79a86
https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
https://setosa.io/ev/image-kernels/

'''