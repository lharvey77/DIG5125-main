# First we import opencv and matplotlib
# (we will use the latter to display our image)
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Now we specify where our image is in relation to this script
# Since it's in the same folder, we write './image_name.jpg'
image_path = './mario.jpg'
# We then use the cv2.imread function to load the image
# from the specified path
image = cv2.imread(image_path)
# OpenCV reads images in BGR instead of RGB by default,
# so we will convert from BGR to RGB so that colours look right
# when we visualise the image below.
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #BGR2GRAY , HSV , RGB , LUV

# ------------------------------------------------------------------------------------------------------
# 3.2 code
# This multiplies the RGB values of each pixel by 5 and subtracts 100
processed_image = image * 5.0 - 100 #1.0 - 1.0 is the normal image. - 1000 is a black screen. 100 - 5.0 is a glitchy swirl
# Now we ensure that no pixel values exceed the range 0-255
# (the allowed range for an 8-bit image)
processed_image = np.clip(processed_image, 0, 255) # changing 255 will change the brightness (eg. 100 is darker)
# The multiplication operation implicitly converts our image from integer
# to floating point values, so we need to convert it back to
# 8-bit integer values, before we display it.
processed_image = processed_image.astype('uint8')
# Now we need to update our visualisation code to show the images side by side
fig, ax = plt.subplots(1,2) # This makes a figure with two subfigures
# In the first subfigure we display the original image
ax[0].imshow(image)
ax[0].set_xlabel('The original image')
ax[0].set_xticks([]) # removes the x axis ticks (so it looks pretty)
ax[0].set_yticks([]) # removes the y axis ticks
# In the second subfigure we display the processed image
ax[1].imshow(processed_image)
ax[1].set_xlabel('The processed image')
ax[1].set_xticks([]) # removes the x axis ticks (so it looks pretty)
ax[1].set_yticks([]) # removes the y axis ticks
plt.show() # Display the figures we just created on-screen
# ------------------------------------------------------------------------------------------------------

# Now that we have our image loaded and we have converted the colour,
# we can display it using:
plt.imshow(image)
plt.title('WOW!') # You can add a title using plt.title()
plt.xlabel('A label for the x axis') # Or axis labels using plt.xlabel()
plt.ylabel('A lable for the y axis') # and plt.ylabel()
plt.show() # Finally we can render the image on-screen

# ------------------------------------------------------------------------------------------------------
# To save an image we use the cv2.imwrite function and pass
# a destination path/filename
# as well as the variable containing our processed image
# Since we're using opencv we'll need to covert from RGB to BGR before saving
#processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
# All done
#cv2.imwrite('./my_amazing_processed_image.png', processed_image)