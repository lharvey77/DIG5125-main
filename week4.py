import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#Task 1 - Channel Check function
def ChannelCheck(imgname):
    imageA = cv2.imread(imgname, cv2.IMREAD_UNCHANGED)

    if imageA is None:
        print("Error: Could not read the image")
        exit()

    dims = np.shape(imageA)

    if len(dims)==3 and dims[2] ==3:
        imagea1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    elif len(dims) == 3 and dims[2] > 3:
        print("Image is not an RGB Image")
    else:
        print('Image A is a Grayscale image, no conversion needed')

    cv2.imshow('imagea1', imagea1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''ChannelCheck('lena.jpg')'''


#Task 2 - Resize images to match
imageA_RGB = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)
imageB_RGB = cv2.imread('mario.jpg', cv2.IMREAD_COLOR)

if imageA_RGB is None or imageB_RGB is None:
    print("Error: Could not read one or both images.")
    exit()

imageA1 = cv2.cvtColor(imageA_RGB, cv2.COLOR_BGR2GRAY)
imageB1 = cv2.cvtColor(imageB_RGB, cv2.COLOR_BGR2GRAY)

sizeA = imageA1.shape
sizeB = imageB1.shape

if sizeA != sizeB:
    print("The images are different sizes. Resizing imageB1 to match ImageA1.")
    imageB1 = cv2.resize(imageB1, (sizeA[1], sizeA[0]))
else:
    print("The images are the same size, therefore I can continue.")

#Threshold for binary
_, imageA2 = cv2.threshold(imageA1, 127, 255, cv2.THRESH_BINARY)
_, imageB2 = cv2.threshold(imageB1, 127, 255, cv2.THRESH_BINARY)


#Task 3 - Merging images
imageC = cv2.bitwise_and(imageA2, imageB2)
imageD = cv2.bitwise_or(imageA2, imageB2)
imageE = cv2.bitwise_not(imageA2, imageB2)

fig = plt.figure()
gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 2, 2], height_ratios=[2,1])

ax1 = plt.subplot(gs[0])
ax1.imshow(cv2.cvtColor(imageA_RGB, cv2.COLOR_BGR2RGB))
ax1.set_title('original A')
ax1.axis('off')

ax2 = plt.subplot(gs[1])
ax2.imshow(imageA1, cmap='gray')
ax2.set_title('grayscale')
ax2.axis('off')

ax3 = plt.subplot(gs[2])
ax3.imshow(imageA2, cmap='gray')
ax3.set_title('Binary')
ax3.axis('off')

ax4 = plt.subplot(gs[5])
ax4.imshow(cv2.cvtColor(imageB_RGB, cv2.COLOR_BGR2RGB))
ax4.set_title('original B')
ax4.axis('off')

ax5 = plt.subplot(gs[6])
ax5.imshow(imageB1, cmap='gray')
ax5.set_title('grayscale')
ax5.axis('off')

ax6 = plt.subplot(gs[7])
ax6.imshow(imageB2, cmap='gray')
ax6.set_title('BW')
ax6.axis('off')

ax7 = plt.subplot(gs[3])
ax7.imshow(imageC, cmap = 'gray')
ax7.set_title('AND Image')
ax7.axis('off')

ax7 = plt.subplot(gs[4])
ax7.imshow(imageD, cmap = 'gray')
ax7.set_title('OR Image')
ax7.axis('off')

ax7 = plt.subplot(gs[8])
ax7.imshow(imageE, cmap = 'gray')
ax7.set_title('NOT Image')
ax7.axis('off')

plt.tight_layout()
plt.show()

#Task 4 - Check dimensions, convert to grayscale, convert to binary. Store in MyImageBW
image4 = cv2.imread('lena.jpg', cv2.IMREAD_COLOR)
size4 = image4.shape
print(size4[1], size4[0])
bw = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
_, MyImageBW = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('bw', bw)
cv2.imshow('MyImageBW', MyImageBW)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Task 5 - Apply dilation and then erosion
MyImageBW = np.copy(imageA2)

MyStrel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

MyDilation = cv2.dilate(MyImageBW, MyStrel, iterations=1)
MyErosion = cv2.erode(MyImageBW, MyStrel, iterations=1)

images = [cv2.cvtColor(imageA_RGB, cv2.COLOR_BGR2RGB), MyImageBW, MyDilation, MyErosion, MyStrel]
titles = ['Original', 'BW Image', 'Dilation', 'Erosion', 'My Strel']

plt.figure()

for i, (img, title) in enumerate(zip(images, titles), 1):
    plt.subplot(1,5,i)
    if i == 1:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
plt.tight_layout()
plt.show()

my_strel_image = cv2.imread('mystrel.png', cv2.IMREAD_GRAYSCALE)

_, my_strel_image = cv2.threshold(my_strel_image, 127, 255, cv2.THRESH_BINARY)

my_image_bw = cv2.imread('Roald.png', cv2.IMREAD_GRAYSCALE)
my_dilation = cv2.dilate(my_image_bw, my_strel_image, iterations=1)

cv2.imshow('Dilated Image', my_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()