import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, minimum_filter, maximum_filter

def my_spatial_filter(my_image_name, filterscale, filter_type):

    # Read the image
    I = plt.imread(my_image_name)
    
    # If the image is RGB, convert it to grayscale
    if I.ndim == 3:
        I = np.mean(I, -1)

    # Show the original image
    plt.imshow(I, cmap='gray')
    plt.title('Original Image')
    plt.show()

    my_filtered_image = np.zeros_like(I)

    # Define a mask size for the filter (initially 5x5)
        #mask_size = 5
    if filterscale % 2 == 0:
        raise ValueError("Filterscale must be an odd integer.")

    # Padding size
    padd_size = filterscale // 2

    # Pad the image with zeros
    I_padded = np.pad(I, ((padd_size, padd_size), (padd_size, padd_size)), mode='symmetric')

    # Show the padded image
    plt.imshow(I_padded, cmap='gray')
    plt.title('Padded Image')
    plt.show()

    # Loop through the image using nested loops extracting the pixel region
    for i in range(padd_size, I_padded.shape[0] - padd_size):
        for j in range(padd_size, I_padded.shape[1] - padd_size):
            pixbuffer = I_padded[i - padd_size:i + padd_size + 1, j - padd_size:j + padd_size + 1]
            if filter_type == 'mean':
                my_filtered_image[i - padd_size, j - padd_size] = np.mean(pixbuffer)
            elif filter_type == 'median':
                my_filtered_image[i - padd_size, j - padd_size] = np.median(pixbuffer)
            elif filter_type == 'min':
                my_filtered_image[i - padd_size, j - padd_size] = np.min(pixbuffer)
            elif filter_type == 'max':
                my_filtered_image[i - padd_size, j - padd_size] = np.max(pixbuffer)
            elif filter_type == 'range':
                my_filtered_image[i - padd_size, j - padd_size] = np.max(pixbuffer) - np.min(pixbuffer)
            else:
                raise ValueError("Choose 'mean', 'median', 'min', 'max', or 'range'")

    # Display the filtered image
    plt.imshow(my_filtered_image, cmap='gray')
    plt.title('Filtered Image')
    plt.show()
    return my_filtered_image

# Example of how to use the function
filtered_image = my_spatial_filter("SaltPeppernoise.png", filterscale=5, filter_type='median')