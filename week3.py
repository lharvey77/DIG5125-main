import cv2
import numpy as np
import matplotlib.pyplot as plt


#Grayscale
#image = cv2.imread('pout.jpg', 0)


#hist = cv2.calcHist([image], [0], None, [256], [0, 256])
#x_values = np.arange(256)


#plt.bar(x_values, hist.ravel(), color='gray')
#plt.title("Grayscale Histogram")
#plt.xlabel("Pixel Value")
#plt.ylabel("Frequency")
#plt.show()


#---------------------------------------------------------------------------------------


#Built-in Function
#image = cv2.imread('lena.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


#hist_r = cv2.calcHist([image], [0], None, [256], [0,256])
#hist_g = cv2.calcHist([image], [1], None, [256], [0,256])
#hist_b = cv2.calcHist([image], [2], None, [256], [0,256])
#x_values = np.arange(256)


#plt.figure()
#plt.bar(x_values, hist_r.ravel(), color='red', alpha=0.5, label='Red channel')
#plt.bar(x_values, hist_g.ravel(), color='green', alpha=0.5, label='Green channel')
#plt.bar(x_values, hist_b.ravel(), color='blue', alpha=0.5, label='Blue channel')
#plt.title("RGB Histogram")
#plt.xlabel("Pixel Value")
#plt.ylabel("Frequency")
#plt.legend()
#plt.show()


#---------------------------------------------------------------------------------------

#DIY Histogram Function

#array_2d = np.array([[1, 2, 3],
#                    [4, 5, 6]])

#array_1d = array_2d.ravel()

#print("Original 2D array")
#print(array_2d)
#print("Flattened 1D array")
#print(array_1d)

#Zeroes

#array_1d = np.zeros(5)
#array_2d = np.zeros((3,4))
#array_3d = np.zeros((2, 3, 4))
#array_2d_int = np.zeros((3, 4), dtype=int)

#Create empty grayscale image of 512x512 pixels
#empty_image = np.zeros((512, 512), dtype=np.uint8)
#Create empty rgb image of 512x512 pixels
#empty_rgb_image = np.zeros((512, 512, 3), dtype=np.uint8)

#---------------------------------------------------------------------------------------

#DIY Histogram
#def grayscale_histogram(image):
    #hist = np.zeros(256, dtype=int)
    #for pixel in image.ravel():
    #    hist[pixel] += 1
    #return hist

#image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
#hist = grayscale_histogram(image)
#plt.plot(hist)
#plt.show()

#Modify this code to dispay the hist data with a bar plot instead of a line plot.

#def grayscale_histogram(image):
    #hist = np.zeros(256, dtype=int)
    #for pixel in image.ravel():
    #    hist[pixel] += 1
    #return hist

#image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
#hist = grayscale_histogram(image)

#plt.bar(range(256), hist, width=1, color='black')
#plt.xlim([0, 255])
#plt.xlabel('Pixel Value')
#plt.ylabel('Frequency')
#plt.title('Grayscale Histogram')
#plt.show()

#Create your own RGB histogram function
#image = cv2.imread('lena.jpg')
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


#hist_r = cv2.calcHist([image], [0], None, [256], [0,256])
#hist_g = cv2.calcHist([image], [1], None, [256], [0,256])
#hist_b = cv2.calcHist([image], [2], None, [256], [0,256])
#x_values = np.arange(256)


#plt.figure()
#plt.bar(x_values, hist_r.ravel(), color='red', alpha=0.5, label='Red channel')
#plt.bar(x_values, hist_g.ravel(), color='green', alpha=0.5, label='Green channel')
#plt.bar(x_values, hist_b.ravel(), color='blue', alpha=0.5, label='Blue channel')
#plt.title("RGB Histogram")
#plt.xlabel("Pixel Value")
#plt.ylabel("Frequency")
#plt.legend()
#plt.show()

#---------------------------------------------------------------------------------------
#Theory and formula

#image = cv2.imread('pout.jpg', cv2.IMREAD_GRAYSCALE)
#equalized_image = cv2.equalizeHist(image)
#plt.imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
#plt.show()

#--------------------------------------

#Using a 1x2 subplot, display the low contrast “pout.tiff” image next to the corresponding grayscale histogram.

#image = cv2.imread('pout.jpg', cv2.IMREAD_GRAYSCALE)
#hist = cv2.calcHist([image], [0], None, [256], [0, 256])
#fig, ax = plt.subplots(1, 2, figsize=(12, 5))

#ax[0].imshow(image, cmap='gray')
#ax[0].set_title('Low Contrast Image')
#ax[0].axis('off') 
#ax[1].plot(hist)
#ax[1].set_title('Grayscale Histogram')
#ax[1].set_xlim([0, 256]) 


#plt.tight_layout()
#plt.show()

#--------------------------------------

#Apply histogram equalisation using the method provided, and dis￾play the new output and new histogram on another 1 x 2 subplot and observe the differences.

#image = cv2.imread('pout.jpg', cv2.IMREAD_GRAYSCALE)
#equalized_image = cv2.equalizeHist(image)

#hist = cv2.calcHist([image], [0], None, [256], [0, 256])
#fig, ax = plt.subplots(1, 2, figsize=(12, 5))

#ax[0].imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
#ax[0].set_title('Equalised Image')
#ax[0].axis('off') 
#ax[1].plot(hist)
#ax[1].set_title('Grayscale Histogram')
#ax[1].set_xlim([0, 256]) 

#plt.tight_layout()
#plt.show()

#---------------------------------------------------------------------------------------
#Formula using built-in function
image = cv2.imread('pout.jpg', cv2.IMREAD_GRAYSCALE)
normalized_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

plt.imshow(cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB))
plt.show()

#Using the formula provided create a historgram normalisation function
#def histogram_normalisation(image,):
    #img_normalized = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    