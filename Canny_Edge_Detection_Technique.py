import cv2
import numpy as np

Road_Image = cv2.imread('Road_Image.jpg')  # Reads image in the form of a numpy array.

# Canny edge detection:-

'''Copying image array rather than assigning is the good practice
because in this way it doesnt change original image'''
road_image_copy = np.copy(Road_Image)

'''Grayscale image only has one channel & each pixel with only one intensity value ranging from 0 to 255.
Hence, Converting to greyscale ,image processing a single channel is faster than processing 3 channel image
and less computational intensive'''
gray_image = cv2.cvtColor(road_image_copy,cv2.COLOR_RGB2GRAY)

'''Image Noise can create false edges & ultimately affect edge detection.
Thus, Applying Guassian Blur by convolving our image with a 5*5 kernel of Guassian values reduces noise
in our image. It Smoothens the image with a Guassian Filter (averaging out the pixels in the image)'''
filtered_image = cv2.GaussianBlur(gray_image,(5,5),0)


'''Edges are the regions where there is a sharp change in intensity or a sharp change in color between
adjacent pixels in the image which will be either Strong Gradient (Steep Change) or Small Gradient (Shallow change).
Canny computes the gradient in all directions of our blurred image and trace the Strongest Gradient as a series of
white pixels'''
canny_image = cv2.Canny(filtered_image,30,90) # Threshold ratio as prescribed 1:3 or 1:2

cv2.imshow("Image",canny_image)
cv2.waitKey(0)  # Keep the image as long as we don't press any key from keyboard.
cv2.destroyAllWindows()
cv2.imwrite('Canny_Image.png',canny_image)  # To save the Canny detected image.
