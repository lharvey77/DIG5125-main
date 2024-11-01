import cv2 
  
# Reads the image 
img = cv2.imread('mario.png') 
  
  
laplacian = cv2.Laplacian(img, cv2.CV_64F) 
cv2.imshow('EdgeMap', laplacian)  
  
cv2.waitKey(0)          
cv2.destroyAllWindows() 