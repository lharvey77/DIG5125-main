import cv2 #importing OpenCV for video altering
import numpy as np #importing Numpy for maths
import matplotlib.pyplot as plt #importing matplotlib for plotting window to display photo(s)
from matplotlib.widgets import Button #importing ButtonEvent for additional functionality

filepath = input("Enter filepath of image you wish to use.")
img = cv2.imread(filepath)
if img is None:
    print("Filepath not valid.")
    exit()

blur_enabled = False
edge_enabled = False
sharpen_enabled = False
blur_enabled_v = False
blur_enabled_h = False

def update_plot():
    global img_final
    img_final = apply_all_eff(img)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Image after effect')
    
    plt.draw()

def apply_all_eff(img):
    global img_result
    img_result = img.copy()
    
    kernel_s = 30
    kernel_vert = np.zeros((kernel_s, kernel_s))
    kernel_hori = np.copy(kernel_vert)
    kernel_vert[:, int((kernel_s - 1)/2)] = np.ones(kernel_s)
    kernel_hori[int((kernel_s - 1)/2), :] = np.ones(kernel_s)
    kernel_vert /= kernel_s
    kernel_hori /= kernel_s
    if blur_enabled_v:
        vert_k = cv2.filter2D(img, -1, kernel_vert)
        img_result = vert_k
    if blur_enabled_h:
       hori_k = cv2.filter2D(img, -1, kernel_hori)
       img_result = hori_k
        
    if edge_enabled:
        edg_pre = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(edg_pre, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(edg_pre, cv2.CV_64F, 0, 1, ksize=5)
        img_result_gr = (sobel_x**2 + sobel_y**2)**0.25
        img_result = cv2.cvtColor(img_result_gr.astype(np.uint8), cv2.COLOR_GRAY2RGB)  

    if sharpen_enabled:
        sh_kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        img_result = cv2.filter2D(src=img_result, ddepth=-1, kernel=sh_kernel)
        
    return img_result
         
def apply_vertical(event):
    global blur_enabled_v
    blur_enabled_v = not blur_enabled_v
    print("Vertical motion Blur", "Enabled" if blur_enabled_v else "Disabled")
    update_plot()
    
def apply_horizontal(event):
    global blur_enabled_h
    blur_enabled_h = not blur_enabled_h
    print("Horizontal motion Blur", "Enabled" if blur_enabled_h else "Disabled")
    update_plot()
    
def apply_edgedet(event):
    global edge_enabled
    edge_enabled = not edge_enabled
    print("Edge Detection", "Enabled" if edge_enabled else "Disabled")
    update_plot()
def apply_sharpen(event):
    global sharpen_enabled
    sharpen_enabled = not sharpen_enabled
    print("Sharpening", "Enabled" if sharpen_enabled else "Disabled")
    update_plot()
    
def save_image(event):
    save_path = input("Enter name for saved image. Please include .jpg or .png")
    cv2.imwrite(save_path, img)
    
bl_button_v = plt.axes([0.7, 0.01, 0.25, 0.075]) 
blurbuttonb = Button(bl_button_v, "Toggle Vertical Blur")
blurbuttonb.on_clicked(apply_vertical)

bl_button_h = plt.axes([0.7, 0.1, 0.25, 0.075]) 
blurbuttonh = Button(bl_button_h, "Toggle Horizontal Blur")
blurbuttonh.on_clicked(apply_horizontal)

edg_button = plt.axes([0.4, 0.01, 0.25, 0.075])
edgebutton = Button(edg_button, "Toggle Edge Detection")
edgebutton.on_clicked(apply_edgedet)

shrp_button = plt.axes([0.1, 0.05, 0.25, 0.075])
shrpbutton = Button(shrp_button, "Toggle Sharpening")
shrpbutton.on_clicked(apply_sharpen)

save_button = plt.axes([0.4, 0.1, 0.25, 0.075]) 
savebutton = Button(save_button, "Save image")
savebutton.on_clicked(save_image)

update_plot()
plt.show()