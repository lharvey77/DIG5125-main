import numpy as np
import matplotlib.pyplot as plt
def my_spatial_filter(my_image_name, filterscale):

    #This is a simple function for calculating a spatial filter
    #on an image, here we will start with a simple mean filter.
    #Parameters:
    #- my_image_name: Path to a grayscale image.
    #Returns:
    #- my_filtered_image: Resultant filtered image.

    # Read the image
    I = plt.imread(my_image_name)
    
    # If the image is RGB, convert it to grayscale
    if I.ndim == 3:
        I = np.mean(I, -1)

    # Show the original image
    plt.imshow(I, cmap='gray')
    plt.title('Original Image')
    plt.show()

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

    # Create an output array for the filtered image data
    I2 = np.zeros_like(I_padded)

    # Loop through the image using nested loops extracting the pixel region
    for i in range(padd_size, I_padded.shape[0]-padd_size):
        for j in range(padd_size, I_padded.shape[1]-padd_size):
            pixbuffer = I_padded[i-padd_size:i+padd_size+1, j-padd_size:j+padd_size+1]
            I2[i, j] = np.mean(pixbuffer)

    # Extract the valid region from the filtered image
    my_filtered_image = I2[padd_size:-padd_size, padd_size:-padd_size]

    # Display the filtered image
    plt.imshow(my_filtered_image, cmap='gray')
    plt.title('Filtered Image')
    plt.show()
    return my_filtered_image

# Example of how to use the function
filtered_image = my_spatial_filter("Roald.png", filterscale=15)