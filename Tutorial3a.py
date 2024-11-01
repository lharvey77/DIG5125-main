import cv2
import matplotlib.pyplot as plt
import numpy as np

#read the grayscale image. Including '0' as an input parameter converts the image to grayscale

img = cv2.imread('Images/lena.tiff',0)


# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

# [img] is the image we wish to apply the histogram

# the second parameter is the channel we wish to apply the histogram to. If using a gray image
# there is only one channel, so we select 0. for a color image R:0 G:1 B:2

# The third input parameter is the mask image, this is for if we wish to apply a histogram to a portion
# of the image. 

# The forth input parameter is the bin the BIN count, for full scale use 256

# The fifth is the range, normally it is 0-256

hist = cv2.calcHist([img], [0], None,[256],[0,256])

# Now we need to create a plot for our histogram data
# We can create values for the X-axis 0-255 using np.arrange()

x_values = np.arange(256)

# Then use pyplot to plot the outputon a bar plot
# Ravel flattens an array eg:
#arr = np.array([[1, 2, 3], [4, 5, 6]])
# print(arr.ravel())
#[1 2 3 4 5 6]
# For a grayscale image, where you typically compute a single histogram for intensity values, 
# the shape of the output histogram would be (num_bins, 1), where num_bins is typically 256 
# the output shape will be (256, 1). this is a 2D array, the plot function converts it to a 1D array

plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title('Grayscale Image')
plt.subplot(1,2,2)
plt.bar(x_values, hist.ravel(), color='gray')
plt.title('Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


## Now I repeat the proces for an RGB image
img2 = cv2.imread('Images/mario.jpg')
img_rgb =  cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

hist_gray = cv2.calcHist([img_gray], [0], None,[256],[0,256])
hist_R = cv2.calcHist([img_rgb], [0], None,[256],[0,256])
hist_G = cv2.calcHist([img_rgb], [1], None,[256],[0,256])
hist_B = cv2.calcHist([img_rgb], [2], None,[256],[0,256])

plt.figure()
plt.subplot(1,2,1)
plt.imshow(img_rgb)
plt.title('RGB Image')
plt.subplot(1,2,2)
plt.bar(x_values,hist_R.ravel(),color='red', alpha=0.5,label='Red Channel')
plt.bar(x_values,hist_G.ravel(),color='green', alpha=0.5,label='Green Channel')
plt.bar(x_values,hist_B.ravel(),color='blue', alpha=0.5,label='Blue Channel')
plt.title('RGB Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()


#Q1 Create a 2x2 subplot that shows a histogram for each of the colour channels and the grayscale histogram.

plt.figure()
plt.subplot(2,2,1)
plt.bar(x_values, hist_gray.ravel(), color='gray')
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(2,2,2)
plt.bar(x_values,hist_R.ravel(),color='red')
plt.title('Red Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(2,2,3)
plt.bar(x_values,hist_G.ravel(),color='green')
plt.title('Green Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(2,2,4)
plt.bar(x_values,hist_B.ravel(),color='blue')
plt.title('Blue Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.tight_layout()

plt.show()


# Q2 Modify this code to dispay the hist data with a bar plot instead

def grayscale_histogram(image):
    hist = np.zeros(256, dtype=int)
    for pixel in image.ravel():
        hist[pixel] += 1
    return hist

image2 = cv2.imread('Images/mandril.tif', cv2.IMREAD_GRAYSCALE)
hist_mandril = grayscale_histogram(image2)

plt.plot(hist_mandril)
plt.title('line plot')
plt.show()

plt.bar(x_values, hist_mandril.ravel(), color='gray')
plt.title('bar plot')
plt.show()

# Histogram Equalisation

# Read in low contrast pout image

image_pout = cv2.imread('Images/pout.tif',cv2.IMREAD_GRAYSCALE)
equalized_image = cv2.equalizeHist(image_pout)

plt.subplot(1,2,1)
plt.imshow(image_pout, cmap='gray',vmin=0, vmax=255)
plt.title('original')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(equalized_image, cmap='gray',vmin=0, vmax=255)
plt.title('Equalised')
plt.axis('off')
plt.tight_layout()
plt.show()

# Histogram Normalisation input parameters
# src: The source image or array which you want to normalize.

# dst: The destination (output) array or image of the same size and type as src. 
# In many use cases, you can set it to None, and OpenCV will automatically create the destination image for you.

# alpha: This is the lower bound of the normalized range of intensities. 
# In normalization to a range, it represents the lower bound. 
# In normalization using a norm, it serves as the norm value.

# beta: This is the upper bound of the normalized range of intensities. 
# This parameter is used only for range normalization.

#norm_type: Defines the type of normalization. Commonly used normalization types include:
#cv2.NORM_INF
#cv2.NORM_L1
#cv2.NORM_L2
#cv2.NORM_MINMAX (for range normalization)

#Here the one we want is a simple range normalisation
normalised_image = cv2.normalize(image_pout, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)


plt.subplot(1,2,1)
plt.imshow(image_pout, cmap='gray',vmin=0, vmax=255)
plt.title('original')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(normalised_image,cmap='gray',vmin=0, vmax=255)
plt.title('Range Normalised')
plt.axis('off')
plt.tight_layout()
plt.show()

#Q3 Using the formula provided create a historgram normalisation function:
def histogram_normalization(image, a=0, b=255):
    Imin = np.min(image)
    Imax = np.max(image)
    output = np.round(a+((image-Imin)*((b-a)/(Imax-Imin))))
    return np.clip(output, a, b).astype('uint8')

my_normalised_image = histogram_normalization(image_pout,0,255)
Imin2 = my_normalised_image.min()
Imax2 = my_normalised_image.max()
print(f"Imin",Imin2,"Imax",Imax2)
plt.subplot(1,2,1)
plt.imshow(image_pout, cmap='gray',vmin=0, vmax=255)
plt.title('original')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(my_normalised_image,cmap='gray',vmin=0, vmax=255)
plt.title(' Normalised')
plt.axis('off')
plt.tight_layout()
plt.show()