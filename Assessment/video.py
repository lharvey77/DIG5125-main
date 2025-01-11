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

kernel = np.ones((5, 5), np.uint8) #kernel is used on lines 110 and 111 for 

def preproc_frame(frame): #a function to prepare every frame for object tracking and effects (pre process)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #turns the frame black and white
    blur = cv2.GaussianBlur(gray, (21, 21), 0) #gaussian blurs the frame
    return blur #returns the b/w blurred frame

filepath = input("Enter the path of your file. Include the filetype (EG .mp4). Enter '0' if you are using a webcam.") #Allows the user to input their own filepath
cap = cv2.VideoCapture(filepath) #assigns the frame capture the value given by the user
if not cap.isOpened(): #if the cap can't be opened,
    print("Error reading video camera or file.") #return error message
    exit() #exit program

fps = cap.get(cv2.CAP_PROP_FPS) #use cv2 to retrieve the frames per second

ret, first_frame = cap.read() #"ret" returns if the frame is available, "first_frame" is returning the first/current frame = the video cap is read
prev_frame = preproc_frame(first_frame) #the frame is updated to be pr-processed

track_paths = [] #an array for the paths to visualise tracking
selected_obj = None #a variable for the objects being tracked. this is used for 
current_centroids = []
blur_enabled = False #variable for enabling or disabling Gaussian blur filter
edge_enabled = False #variable for enabling or disabling Edge Detection

def get_centroid(contour):
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return cx, cy
    return None

def visual_tracking_path(frame, object_index): #function for visualising the objects tracking path
    if object_index is not None: #as long as there is an object to track,
        for path in track_paths[object_index]: #for every movement/path in the track_paths array,
            cv2.circle(frame, path, 3, (0, 0, 255), -1) #draw a circle in every movement/path location

maximum_points = 50 #to prevent the screen from filling with too many circles, the tracking path will delete the oldest circle past 50

fig, ax= plt.subplots(figsize=(12, 8)) #Creating display with matplotlib, with a size declared so it's bigger than the smaller default size
plt.axis("off") #turn off the axis because it's for a video, not a graph

def on_click(event: MouseEvent): #a function for clicking on the matplotlib display
    global selected_obj #
    if event.inaxes != ax:
        return
    
    click_x, click_y = int(event.xdata), int(event.ydata)
    min_distance = float('inf')
    
    closest_obj_index = None
    for i, centroid in enumerate(current_centroids):
        distance = np.sqrt((centroid[0] - click_x)**2 + (centroid[1] - click_y)**2)
        if distance < min_distance:
            min_distance = distance
            closest_obj_index = i

    if closest_obj_index is not None:
        selected_obj = closest_obj_index
        print(f"Selected object {selected_obj + 1}")

def apply_blur(event):
    global blur_enabled
    blur_enabled = not blur_enabled  # Toggle blur state
    print("Gaussian Blur", "Enabled" if blur_enabled else "Disabled")
    
def apply_edgedet(event):
    global edge_enabled
    edge_enabled = not edge_enabled
    print("Edge Detection", "Enabled" if edge_enabled else "Disabled")
    
bl_button = plt.axes([0.7, 0.05, 0.2, 0.075]) 
blurbutton = Button(bl_button, "Toggle Gaussian Blur")
blurbutton.on_clicked(apply_blur)

edg_button = plt.axes([0.3, 0.05, 0.2, 0.075])
edgebutton = Button(edg_button, "Toggle Edge Detection")
edgebutton.on_clicked(apply_edgedet)

fig.canvas.mpl_connect("button_press_event", on_click)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading video camera or file.")
        break

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
    cv2.putText(frame, "Click to select an object", (10, 1030), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
