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
kernel = np.ones((5, 5), np.uint8)

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
            
def apply_blur(event):
    global blur_enabled #use the blur_enabled as a global variable so it can be used/seen outside the function (this will trigger the if statement in the main loop)
    blur_enabled = not blur_enabled  #to toggle the blur state
    print("Gaussian Blur", "Enabled" if blur_enabled else "Disabled") #print a message in the terminal whether the blur is enabled or not 
    
def apply_edgedet(event):
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

    fig.canvas.manager.set_window_title("Video - Object Tracking")
    current_frame = preproc_frame(frame)

    frame_diff = cv2.absdiff(prev_frame, current_frame)
    thresh = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_centroids.clear()

    for contour in contours:
        if cv2.contourArea(contour) < 800:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)

        centroid = get_centroid(contour)
        if centroid:
            if len(current_centroids) < 9:
                current_centroids.append(centroid)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                object_number = len(current_centroids)
                cv2.putText(frame, str(object_number), (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    while len(track_paths) < len(current_centroids):
        track_paths.append([])

    for i, obj in enumerate(current_centroids):
        if i < 9:
            track_paths[i].append(obj)
            if len(track_paths[i]) > maximum_points:
                track_paths[i].pop(0)

    if selected_obj is not None:
        visual_tracking_path(frame, selected_obj)

    if blur_enabled:
        for contour in contours:
            if cv2.contourArea(contour) < 800:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            blurred_region = cv2.GaussianBlur(frame[y:y+h, x:x+w], (99, 99), 30)
            frame[y:y+h, x:x+w] = blurred_region
            
    if edge_enabled:
        for contour in contours:
            if cv2.contourArea(contour) < 800:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            edge_region = cv2.Canny(frame, 10, 70)
            ret, frame = cv2.threshold(edge_region, 70, 255, cv2.THRESH_BINARY)

    object_count = len(current_centroids)
    cv2.putText(frame, f"Objects counted: {object_count}", (10, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "Click on object to visualise tracking.", (10, 1030), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Tracking {selected_obj + 1 if selected_obj is not None else 'None'}")
    plt.pause(1 / fps)

    key = cv2.waitKey(int(1000 / fps)) & 0xFF
    if key == ord('q'):
        break

    prev_frame = current_frame

cap.release()
cv2.destroyAllWindows()


'''
https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
https://byjus.com/maths/centroid/
https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
https://en.codingwalks.com/entry/OpenCV-Python-Contour-Detection-and-Labeling-1
https://matplotlib.org/stable/users/explain/figure/event_handling.html
https://byjus.com/maths/euclidean-distance/
https://matplotlib.org/stable/users/explain/figure/event_handling.html
https://www.futurelearn.com/info/courses/introduction-to-image-analysis-for-plant-phenotyping/0/steps/305359
'''