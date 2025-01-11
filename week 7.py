import cv2
import numpy as np
import matplotlib.pyplot as plt

'''# Load the image
image = cv2.imread('Roald.png', cv2.IMREAD_GRAYSCALE)

# Define the Roberts operator kernels
roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)

# Apply the Roberts operator
edge_x = cv2.filter2D(image, -1, roberts_x)
edge_y = cv2.filter2D(image, -1, roberts_y)

# Calculate the edge magnitude
edge_magnitude = np.sqrt(np.square(edge_x) + np.square(edge_y))
edge_magnitude = np.uint8(edge_magnitude)

# Display the results
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(edge_magnitude, cmap='gray'), plt.title('Edge Image,(Roberts Operator)')
plt.show()'''

#------------------------------------------------------------------------------------------------------------

'''image = cv2.imread('mario.jpg', cv2.IMREAD_GRAYSCALE)
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
sobel_c = np.sqrt(sobel_x**2 + sobel_y**2) #mine
sobel_c = np.uint8(np.absolute(sobel_c)) #mine
cv2.imshow('Original', image)
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.imshow('Sobels combined', sobel_c)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

#------------------------------------------------------------------------------------------------------------

# Apply Canny edge detection with cv2.canny. The input parameters are the image,the lower threshold and upper threshold
'''edges = cv2.Canny(image, 50, 150)
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

#------------------------------------------------------------------------------------------------------------

'''blurred = cv2.GaussianBlur(image, (5, 5), 0)
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
cv2.imshow('LoG Edge Detection', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

#------------------------------------------------------------------------------------------------------------

'''kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) 
vertical_edges = cv2.filter2D(image, -1, kernel)
cv2.imshow('Horizontal Edges', vertical_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

#------------------------------------------------------------------------------------------------------------

image = cv2.imread('mario.jpg')

def region_segmentation(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 75, 110)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    segmented_image = image.copy()
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 2)
    return segmented_image
def main():

# Load the image
    image = cv2.imread('mario.jpg')
    if image is None:
        print("Could not open or find the image.")
    exit()

# Perform region segmentation
segmented_image = region_segmentation(image.copy())

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title('Region Segmented Image')
plt.show()
if __name__ == "__main__":
    main()