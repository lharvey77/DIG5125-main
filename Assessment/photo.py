import cv2 #importing OpenCV for video altering
import numpy as np #importing Numpy for maths
import matplotlib.pyplot as plt #importing matplotlib for plotting window to display photo(s)
from matplotlib.widgets import Button #importing ButtonEvent for additional functionality
'''Read image'''
filepath = input("Enter filepath of image you wish to use.") #a variable with user input for the  images filepath
img = cv2.imread(filepath) #use cv2 to read the filepath into the image variable
if img is None: #if there is no valid filepath or file,
    print("Filepath not valid.") #tell the user that it is not valid
    exit() #and quit the program

'''Variables'''
blur_enabled = False #Variable for enabling/disabling Gaussian Blur
edge_enabled = False #Variable for enabling/disabling Edge Detection
sharpen_enabled = False #Variable for activating either blur_enabled_v or blur_enabled_h
blur_enabled_v = False #Variable for enabling/disabling vertical motion blur
blur_enabled_h = False #Variable for enabling/disabling horizontal motion blur

'''Functions'''
def update_plot(): #function for updating matplotlib plot
    global img_final #global variable for the layered, edited image
    img_final = apply_all_eff(img) #call on apply_all_eff function to render the final image from the current image
    plt.subplot(1, 2, 1) #create a plot that has 1 row and 2 columns. this is item 1
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #in this item, show the image as an RGB image
    plt.axis('off') #turn off the axis as this is not a graph
    plt.title('Original Image') #title for this plot: this is the original uplaoded image

    plt.subplot(1, 2, 2) #create a plot that has 1 row and 2 columns. this is item 2
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)) #in this item, show the image as an RGB image
    plt.axis('off') #turn off the axis as this is not a graph
    plt.title('Image after effect') #title for this plot: this is the image with effects on
    
    plt.draw() #at the end of the function, draw the plots to render everything

def apply_all_eff(img): #a function for applying multiple effects at once
    global img_result #global variable for the result of effects
    img_result = img.copy() #copy the original image at the start and put it in the variable
    
    '''Setup for motion blur'''
    #a kernel is a matrix used to apply filters to images
    kernel_vert = np.zeros((30, 30)) #this is the kernel for the vertical motion blur. np.zeros will fill the kernel with zeros. 30,30 means it is a 30 by 30 grid
    kernel_hori = np.copy(kernel_vert) #this is the kernel for the horizontal motion blur. it is a copy of the vertical kernel
    kernel_vert[:, 30 // 2] = 1 #[:, 30 // 2] = finding the middle column. =1 = setting the middle column to 1
    kernel_hori[30 // 2, :] = 1 #[30 // 2, :] = finding the middle row. =1 =setting the middle row to 1
    kernel_vert /= 30 #normalize the kernel values
    kernel_hori /= 30 #normalize the kernel values
    if blur_enabled_v: #if the vertical motion blur is enabled,
        vert_k = cv2.filter2D(img, -1, kernel_vert) #make filtered image. img = source. -1 = ddepth (-1 will make it same as source image). kernel_vert = kernel being used
        img_result = vert_k #the image result is now the image that has been filtered with the vertical kernel
    if blur_enabled_h: #if the horizontal motion blur is enabled,
       hori_k = cv2.filter2D(img, -1, kernel_hori) #make filtered image. img = source. -1 = ddepth (-1 will make it same as source image). kernel_hori = kernel being used
       img_result = hori_k #the image result is now the image that has been filtered with the horizontal kernel
        
    if edge_enabled: #edge detection
        edg_pre = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY) #the image that is pre-processed (ready to be processed with filter) in black and white. this is the last img_result so the image result is being updated everytime the function is called
        sobel_x = cv2.Sobel(edg_pre, cv2.CV_64F, 1, 0, ksize=5) #detecting edges on the x axis for sobel edge detection. the pre-processed image is the source. CV_64F is the depth of the image (64 bit format to allow negative values for Sobel to use).
                    #1, 0 = derivative/change in pixel intensity; 1 means there is change in the x axis, 0 means there is no change in the y axis. ksize=5 = means that the kernel being used for the sobel filter is a 5x5 grid
        sobel_y = cv2.Sobel(edg_pre, cv2.CV_64F, 0, 1, ksize=5)#detecting edges on the y axis for sobel edge detection. the pre-processed image is the source. CV_64F is the depth of the image (64 bit format to allow negative values for Sobel to use).
                    #0, 1 = derivative/change in pixel intensity; 0 means there is no change in the x axis, 1 means there is a change in the y axis. ksize=5 = means that the kernel being used for the sobel filter is a 5x5 grid
        img_result_gr = (sobel_x**2 + sobel_y**2)**0.25 #the black and white result is rendered (gr for gray). sobel_x and y are put to the power of 2. they are then added together. the result is put to the power of 0.25.
                    #this is the magnitude of the filter. just adding sobel_x and _y results in too much noise on the image
        img_result = cv2.cvtColor(img_result_gr.astype(np.uint8), cv2.COLOR_GRAY2RGB)  #the final image result is rendered. the magnitude is put into uint8 data type and the black and white is turned into RGB, so it can be displayed in the plot

    if sharpen_enabled: #sharpening
        sh_kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]) #the sharpening kernel is manually declared in a 3x3 array. the -1's sharpen the edges
        img_result = cv2.filter2D(src=img_result, ddepth=-1, kernel=sh_kernel) #the final image result is rendered. the source is the last img_result being overwritten to layer the effect ontop.
                    #the ddepth is -1 to match the source image and the kernel used is the sharpening kernel (sh_kernel)
        
    return img_result#at the end of the function, always return the new img_result
         
def apply_vertical(event): #function for enabling vertical motion blur
    global blur_enabled_v #global variable for enabling vertical motion blur
    blur_enabled_v = not blur_enabled_v #to toggle the blur state
    print("Vertical motion Blur", "Enabled" if blur_enabled_v else "Disabled") #print text to terminal to tell the user if the filter is on or off
    update_plot() #call on the update_plot function to redraw the plots, call on the apply_all_eff function and apply the filter corresponding to the button
    
def apply_horizontal(event): #function for enabling horizontal motion blur
    global blur_enabled_h #global variable for enabling horizontal motion blur
    blur_enabled_h = not blur_enabled_h #to toggle the blur state
    print("Horizontal motion Blur", "Enabled" if blur_enabled_h else "Disabled") #print text to terminal to tell the user if the filter is on or off
    update_plot() #call on the update_plot function to redraw the plots, call on the apply_all_eff function and apply the filter corresponding to the button
    
def apply_edgedet(event): #function for enabling edge detection
    global edge_enabled #global variable for enabling edge detection
    edge_enabled = not edge_enabled #to toggle the edge detection state
    print("Edge Detection", "Enabled" if edge_enabled else "Disabled") #print text to terminal to tell the user if the filter is on or off
    update_plot() #call on the update_plot function to redraw the plots, call on the apply_all_eff function and apply the filter corresponding to the button
    
def apply_sharpen(event): #function for enabling sharpening
    global sharpen_enabled #global variable for enabling sharpening
    sharpen_enabled = not sharpen_enabled #to toggle the edge detection state
    print("Sharpening", "Enabled" if sharpen_enabled else "Disabled") #print text to terminal to tell the user if the filter is on or off
    update_plot() #call on the update_plot function to redraw the plots, call on the apply_all_eff function and apply the filter corresponding to the button
    
def save_image(event): #function for saving an image
    save_path = input("Enter name for saved image. Please include .jpg or .png") #the user can write the name that they want the image saved under. it will go in the same folder that the code/program is in
    cv2.imwrite(save_path, img) #cv2 writes the image under the 'save_path' name variable

'''Buttons'''   
bl_button_v = plt.axes([0.7, 0.01, 0.25, 0.075]) #setup for the button. 0.7 is the left position. 0.01 is the bottom position. 0.25 is the buttons width. 0.075 is the buttons height.
blurbuttonb = Button(bl_button_v, "Toggle Vertical Blur") #text that tells the user this button is for vertical blur
blurbuttonb.on_clicked(apply_vertical) #when the button is clicked, the program calls on the apply_vertical function to apply the filter

bl_button_h = plt.axes([0.7, 0.1, 0.25, 0.075]) #setup for the button. 0.7 is the left position. 0.01 is the bottom position. 0.25 is the buttons width. 0.075 is the buttons height.
blurbuttonh = Button(bl_button_h, "Toggle Horizontal Blur") #text that tells the user this button is for horizontal blur
blurbuttonh.on_clicked(apply_horizontal) #when the button is clicked, the program calls on the apply_vertical function to apply the filter

edg_button = plt.axes([0.4, 0.01, 0.25, 0.075]) #setup for the button. 0.4 is the left position. 0.01 is the bottom position. 0.25 is the buttons width. 0.075 is the buttons height.
edgebutton = Button(edg_button, "Toggle Edge Detection") #text that tells the user this button is for edge detection
edgebutton.on_clicked(apply_edgedet) #when the button is clicked, the program calls on the apply_edgedet function to apply the filter

shrp_button = plt.axes([0.1, 0.05, 0.25, 0.075]) #setup for the button. 0.1 is the left position. 0.05 is the bottom position. 0.25 is the buttons width. 0.075 is the buttons height.
shrpbutton = Button(shrp_button, "Toggle Sharpening") #text that tells the user this button is for sharpening
shrpbutton.on_clicked(apply_sharpen) #when the button is clicked, the program calls on the apply_sharpen function to apply the filter

save_button = plt.axes([0.4, 0.1, 0.25, 0.075]) #setup for the button. 0.4 is the left position. 0.01 is the bottom position. 0.25 is the buttons width. 0.075 is the buttons height.
savebutton = Button(save_button, "Save image") #text that tells the user this button is for saving the filtered image
savebutton.on_clicked(save_image) #when the button is clicked, the program calls on the save_image function to save the image

update_plot() #at the start of the code, update the plot
plt.show() #show the plot that has been updated

'''Websites used
https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
https://stackoverflow.com/questions/43392956/explanation-for-ddepth-parameter-in-cv2-filter2d-opencv
https://www.geeksforgeeks.org/sobel-edge-detection-vs-canny-edge-detection-in-computer-vision/
https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
https://towardsdev.com/how-to-perform-edge-detection-using-sobel-x-and-sobel-y-in-cv2-easiest-explanation-83c4a6a56875
https://www.analyticsvidhya.com/blog/2021/08/sharpening-an-image-using-opencv-library-in-python/
https://setosa.io/ev/image-kernels/


'''