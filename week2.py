import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('Roald.png')

#print(np.shape(image))

#gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

#pixel_value = image[50, 50]
#row_values = image[1, :]
#green_channel = image[1, :, 1]
#image[:, :, 0] -= 25

#cv2.imshow('This is amazing', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#plt.imshow(gray_image, cmap='gray')
#plt.show()

#plt.subplot(2, 2, 1)
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#plt.axis('off')
#plt.title('Original Image')

#plt.subplot(2, 2, 2)
#plt.imshow(cv2.cvtColor(binary_image, cv2.COLOR_BGR2RGB))
#plt.axis('off')
#plt.title('Binary Image')

#plt.subplot(2, 2, 3)
#plt.imshow(cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB))
#plt.axis('off')
#plt.title('Grey Roald Image')

#plt.subplot(2, 2, 4)
#plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2HLS))
#plt.axis('off')
#plt.title('Fun Roald Image')

#plt.show()

def graybinary (image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binimg = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    plt.subplot(1, 2, 1)
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.title('Grayscale Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(binimg, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Binary Image')
    plt.show()


graybinary(image)

def bluegreen (image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    green = image
    blue = image
    image[:, :, 0] -= 25
