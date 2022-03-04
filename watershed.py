from PIL import Image, ImageEnhance, ImageSequence
import numpy as np
import cv2
from matplotlib import pyplot as plt

im = Image.open('stack.tif')

for i, page in enumerate(ImageSequence.Iterator(im)):
    page.save("page%d.png" % i)
    #page.show("page%d.png" % i)

img = cv2.imread('page6.png')
print(type(img))
cv2.imshow('image_original',img)

#conversion to gray image
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#blur to reduce high freq noise
blurred = cv2.GaussianBlur(gray, (1, 1), 0)


#Thresholding
(T, threshInv) = cv2.threshold(blurred, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Threshold", threshInv)
print("[INFO] otsu's thresholding value: {}".format(T)) # any pixel greater than T is set to 0, less than T is set to 255


# noise removal
kernel = np.ones((2, 2), np.uint8)
opening = cv2.morphologyEx(threshInv,cv2.MORPH_OPEN,kernel, iterations = 2)
cv2.imshow("Opening_kernel(2,2)_iter:2", opening)



# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
cv2.imshow("Sure Background", sure_bg)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
cv2.imshow('Dist_Transform',dist_transform)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)


# Finding unknown region
sure_fg = np.uint8(sure_fg)
cv2.imshow('Sure_fg',sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imshow('unknown',unknown)
cv2.waitKey(0)


# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
print(markers)

# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
cv2.imshow("Watershed", img)
cv2.waitKey(0)
