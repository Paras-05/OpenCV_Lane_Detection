import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    filtered_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(filtered_image, 30, 90)
    return canny_image


def interested_region(img):
    height = img.shape[0]
    width = img.shape[1]

    """Vertices are specified by observing the Image
       Also, fillPoly function takes only array of polygons thus we need to pass
       in an array form eventhough there is only single polygon"""
    masked_figure = np.array([
                             [(20, height), (width, 650), (550, 200)]
                             ])

    """ zero like Creates a black image of same dimension as that of original Image
        Then we draw the polygon of region of interest on that black image"""
    mask = np.zeros_like(img)


    cv2.fillPoly(mask, masked_figure, 255)

    """Now, To obtain the edges of the lane, we perform "bitwise_and" of masked image
       with canny image so that only edges corresponding to lanes are visible"""
    cropped_image = cv2.bitwise_and(img, mask)
    return cropped_image


road_image = cv2.imread('Road_Image.jpg')
road_image_copy = np.copy(road_image)
canny_image = canny(road_image_copy)
interested_region(canny_image)

cv2.imshow("result", interested_region(canny_image))
cv2.waitKey(0)
cv2.imwrite('Interested Region.png', interested_region(canny_image))
