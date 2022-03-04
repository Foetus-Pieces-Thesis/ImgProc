from PIL import Image, ImageEnhance, ImageSequence
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy import savetxt


def load(filename):
    '''
    Function: load the tif file and saves it as 'pages'
    Input:
    filename: name of the file
    '''
    im = Image.open('filename')

    for i, page in enumerate(ImageSequence.Iterator(im)):
        page.save("page%d.png" % i)
        #page.show("page%d.png" % i)

#Preprocessing
#contrast,brightness,sharpness modification
def modify_contrast(val,modimg):

    '''
    input:
    val: int, A floating point value controlling the enhancement. Factor 1.0 always
     returns a copy of the original image, lower factors mean less color
     (brightness, contrast, etc), and higher values more.
    modimg: image object(PIL) that we want to process
    output:
    modimg: a numpy array representing the image
    '''

    filtercontrast = ImageEnhance.Contrast(modimg)
    modimg = filtercontrast.enhance(val)
    modimg = np.array(modimg) # convert into a numpy array

    return modimg



def modify_brightness(val,modimg):
    '''
    input:
    val: int, A floating point value controlling the enhancement.
    modimg: image object(PIL) that we want to process
    output:
    modimg: a numpy array representing the image
    '''

    filterbright = ImageEnhance.Brightness(modimg)
    modimg = filterbright.enhance(val)
    modimg = np.array(modimg) # convert into a numpy array

    return modimg


def modify_sharpness(val, modimg):
    '''
    input:
    val: int, A floating point value controlling the enhancement.
    modimg: image object(PIL) that we want to process
    output:
    modimg: a numpy array representing the image
    '''

    filtersharp = ImageEnhance.Sharpness(modimg)
    modimg = filtersharp.enhance(val)
    #modimg = np.array(modimg)  # convert into a numpy array

    return modimg


def convert_grayscale(image):
    '''
    Function: conversion to grayscale image
    Input: image
    Output:
    gray: grayscale image
    '''
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return gray

def Gaussian_blur(image, ksize, sigmaX, sigmaY,borderType):
    '''
    -remove noise that approximately follows a Gaussian distribution.
    Input:
    image: input image
    ksize: <class 'tuple'> Gaussian Kernel Size. [height width]. height and width should be odd and can
    have different values. If ksize is set to [0 0], then ksize is computed from sigma values.
    sigmaX : Kernel standard deviation along X-axis.
    sigmaY: Kernel standard deviation along Y-axis. If sigmaY=0, then sigmaX value is taken for sigmaY
    borderType : Specifies image boundaries while kernel is applied on image borders.
    Possible values are : cv.BORDER_CONSTANT cv.BORDER_REPLICATE cv.BORDER_REFLECT cv.BORDER_WRAP
    cv.BORDER_REFLECT_101 cv.BORDER_TRANSPARENT cv.BORDER_REFLECT101 cv.BORDER_DEFAULT cv.BORDER_ISOLATED
    return:
    '''
    blurred = cv2.GaussianBlur(image, ksize, sigmaX, sigmaY, borderType)
    return blurred
def bilateral_filtering(image, d, sigmaColor, sigmaSpace):
    '''
    Input:
    image
    d: Diameter of each pixel neighborhood.
    sigmaColor: Value of \sigma  in the color space. The greater the value, the colors farther to each other will start to get mixed.
    sigmaSpace: Value of \sigma  in the coordinate space. The greater its value, the more further pixels will mix together, given that their colors lie within the sigmaColor range
    :return:
    '''
    bilateral_filt = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    return bilateral_filt

# Edge Detection
#Sobel
def sobel(image, ddepth, dx, dy, ksize):
    '''
    :return:
    '''
    sobel_filt=cv2.Sobel(src, ddepth, dx, dy, ksize)
    return sobel_filt

#Canny
def canny(image,threshold1, threshold2):
    canny_filt = cv2.Canny(image, threshold1, threshold2)
    return canny_filt

# def sharpen(image,ddepth=)
#
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5,-1],
#                        [0, -1, 0]])
#     image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
#     cv2.imshow('Sharpened_Image', image_sharp)

#Body
#load file
# im = Image.open('stack.tif')
# im = im.convert('RGB') # im is now of type <class 'PIL.Image.Image'>
# print(type(im))
# for i, page in enumerate(ImageSequence.Iterator(im)):
#     page.save("page%d.png" % i)
#     # sharpness enhancement
#     sharpness_enhanced_im = modify_sharpness(10, page)  # 2nd argument cant be a ndarray needs to be a PIL.Image.Image
#     sharpness_enhanced_im = np.array(sharpness_enhanced_im)  # convert into a matrix form (numpy array), cv2.imshow only takes in mat
#     #cv2.imshow('sharpness_enhanced_im', sharpness_enhanced_im)
#     page.save("enhanced_page%d.png"% i)
#     # page.show("page%d.png" % i)

#selecting a specific slice
img = cv2.imread('page3.png')
print(type(img))
print('Dimension of image',img.shape)
cv2.imshow('image_original',img)
cv2.waitKey(0)


# # Saving the array in a csv file
# arr_reshaped = img.reshape(img.shape[0], -1) # reshaping the array from 3D matrice to 2D matrice. array values are concatenated horizontally
# #arr_reshaped dimension is (1024, 3072)
# print('Dimension of arr_reshaped',arr_reshaped.shape)
# savetxt('data_original.csv', arr_reshaped, delimiter=',',fmt='%d')
# maxElement = np.amax(arr_reshaped)
# print('Max value in array(original):',maxElement)

#Gray Scaling
# is there a way to have more levels in the color intensities
# what is the largest number , what is the smallest number?
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_image',gray_image)
#plt.hist(gray_image.ravel(),256,[0,256]); #img.ravel() flattens array into a 1D array without changing input type

# Histogram Equalization https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html
img_equalizeHist = cv2.equalizeHist(gray_image)
cv2.imshow('image_original_equalizehist',img_equalizeHist)

#visualizing data using histogram
plt.hist(img_equalizeHist.ravel(),256,[0,256]);
#plt.legend(loc='upper right') add name to hist labels
#plt.show()

# 'suppress' values
img_equalizeHist[(210>img_equalizeHist)] = 0.0
cv2.imshow('image_original_equalizehist_suppressed',img_equalizeHist)

# iterate the function equalizeHist
for i in range(5): #'ideal' is 100
    img_equalizeHist = cv2.equalizeHist(img_equalizeHist)
    plt.hist(img_equalizeHist.ravel(), 256, [0, 256])
    #plt.show()

# fig.add_subplot(rows, columns, 5)
# plt.imshow(opening,cmap='gray', vmin=0, vmax=255)
# plt.title("opening_(4,4)_iter=1")
#
# opening = cv2.morphologyEx(threshInv, cv2.MORPH_OPEN, kernel, iterations=2)
# fig.add_subplot(rows, columns, 6)
# plt.imshow(opening,cmap='gray', vmin=0, vmax=255)
# plt.title("opening_(4,4)_iter=2")

#display plot of subplots
#plt.show()
cv2.imshow('image_original_equalizehist_suppressed_equalized',img_equalizeHist)

# erosion
kernel = np.ones((2, 2), 'uint8')
erode_img = cv2.erode(img_equalizeHist, kernel, iterations=1)
cv2.imshow('Eroded Image', erode_img)
cv2.waitKey(0)

# #convert back to RGB
# backtorgb = cv2.cvtColor(erode_img,cv2.COLOR_GRAY2RGB)
# cv2.imshow('Enhance Image',backtorgb)
# cv2.waitKey(0)


# # Saving the array in a csv file
# arr_reshaped = gray_image.reshape(img.shape[0], -1) # reshaping the array from 3D matrice to 2D matrice. array values are concatenated horizontally
# #arr_reshaped dimension is (1024, 3072)
# print('Dimension of arr_reshaped(gray)',arr_reshaped.shape)
# savetxt('data_grayimage.csv', arr_reshaped, delimiter=',',fmt='%d')
# maxElement = np.amax(arr_reshaped)
# print('Max value in array:',maxElement)

# Image Sharpening
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
cv2.imshow('Sharpened_Image_ddepth=-1_src=img', image_sharp)
cv2.waitKey(0)
image_sharp = cv2.filter2D(src=gray_image, ddepth=-1, kernel=kernel)
cv2.imshow('Sharpened_Image_ddepth=-1_src=gray_image', image_sharp)
cv2.waitKey(0)
image_sharp = cv2.filter2D(src=erode_img, ddepth=-1, kernel=kernel)
cv2.imshow('Sharpened_Image_grayed_ddepth=-1_src=img_equalizeHist', image_sharp)
cv2.waitKey(0)



# #Pre-Processing
# # bilateral_filtered = bilateral_filtering(image, 9, 200, 200)
# Gaussian_blur(sharpness_enhanced_im, (1, 1), 0, 0, cv2.BORDER_DEFAULT)
#
#
# #convert to grayscale image
# img= convert_grayscale(img)
# print(type(img))
#
# # Edge Detection
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, 20)  # Sobel Edge Detection on the X axis
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, 20)  # Sobel Edge Detection on the Y axis
# sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, 20)  # Combined X and Y Sobel Edge Detection
# cv2.imshow('sobelxy',sobelxy)
# cv2.waitKey(0)
#
#Thresholding
(T, threshInv) = cv2.threshold(erode_img, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
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
