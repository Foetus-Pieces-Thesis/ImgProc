from PIL import Image, ImageEnhance, ImageSequence
import numpy as np
import cv2
from matplotlib import pyplot as plt
from numpy import savetxt
import tensorflow as tf
import skimage.io
import skimage.viewer as skview
import png

# from keras.preprocessing.image import load_img
# import warnings


def load(filename):
    '''
    Function: load the tif file and saves individual stacks, at each stack the corresponding RGB channels are saved
    Input:
    filename: name of the file, assumes dimension of 5 X1024 X1024 X3
    '''
    data = skimage.io.imread(filename)
    #print(data.shape)
    for i in range(5):
        datastack = data[i, :, :, :]
        print(datastack.shape)
        skimage.io.imsave('stack%d.png' % i, datastack)
        for t in range(3):
            channel_split = datastack[:, :, t]
            zeros = np.zeros((1024, 1024), dtype=np.int8)
            if t == 0:  # red channel
                channel_split = np.stack((channel_split, zeros, zeros), axis=2)
                print(channel_split.shape)
                skimage.io.imsave("stack%d,channel%d.png" % (i, t), channel_split)
            elif t == 1:  # green channel
                channel_split = np.stack((zeros, channel_split, zeros), axis=2)
                print(channel_split.shape)
                skimage.io.imsave("stack%d,channel%d.png" % (i, t), channel_split)
            else:  # blue channel
                channel_split = np.stack((zeros, zeros, channel_split,), axis=2)
                print(channel_split.shape)
                skimage.io.imsave("stack%d,channel%d.png" % (i, t), channel_split)

    print("Process 'load' completed")

    return None

#Preprocessing


def mod_con_bri(alpha,beta,iter,img):
    '''
    Function: g(x,y)= |alpha * f(x,y) + beta|
        Scales, calculates absolute values, and converts the result to 8-bit.

    Input:
     alpha:double, for contrast(gain)
     beta:double, for brightness(bias)
     iter:int, number of iterations
    Output:
     adjusted: a numpy array

    #Ideal values for contrast and brightness in mod_con_bri, should implement an iterative process
    alpha = 3.5
    beta = 15
    '''
    for i in range(iter):
        adjusted = cv2.convertScaleAbs(img,alpha=alpha, beta=beta)

    return adjusted


def mod_saturation(image,sat_factor):
    '''
    Input:
    image:	RGB image or images. The size of the last dimension must be 3.
    saturation_factor:	float, Factor to multiply the saturation by.
    :return:
    sat_img: Image modified by saturation function
    '''
    sat_img=tf.image.adjust_saturation(image,sat_factor)  #size of last dimension must be 3; processes RGB
    print('Data type of image(after saturation)',img.dtype)
    sat_img=tf.keras.preprocessing.image.img_to_array(sat_img) #convert back to np.array
    print('Data type of image(after conversion to np)',img.dtype)

    return sat_img



def convert_grayscale(image):
    '''
    Function: conversion to grayscale image from RGB
    Input: image
    :return
        gray: grayscale image
    '''
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
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

## Edge Detection
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

# # Edge Detection Example code
# sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, 20)  # Sobel Edge Detection on the X axis
# sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, 20)  # Sobel Edge Detection on the Y axis
# sobelxy = cv2.Sobel(img, cv2.CV_64F, 1, 1, 20)  # Combined X and Y Sobel Edge Detection
# cv2.imshow('sobelxy',sobelxy)
# cv2.waitKey(0)

#Image Sharpening
def sharpen(image,ddepth=-1):

    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    image_sharp = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
    cv2.imshow('Sharpened_Image', image_sharp)

    return image_sharp

def save_csv():
    # Saving the array in a csv file
    arr_reshaped = img.reshape(img.shape[0], -1) # reshaping the array from 3D matrice to 2D matrice. array values are concatenated horizontally
    #arr_reshaped dimension is (1024, 3072)
    print('Dimension of arr_reshaped',arr_reshaped.shape)
    savetxt('data_original.csv', arr_reshaped, delimiter=',',fmt='%d')
    maxElement = np.amax(arr_reshaped)
    print('Max value in array(original):',maxElement)

    return

def erosion(image,kernel_size_x,kernel_size_y,iterations):
    '''
    Documentation: https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
    Function: As the kernel B is scanned over the image, we compute the minimal pixel value overlapped by B
              and replace the image pixel under the anchor point with that minimal value.
              The erosion operation is: dst(x,y)=min(x′,y′):element(x′,y′)≠0src(x+x′,y+y′)
    Input:
    image: numpy array
    kernel_size: int
    iterations: int, no of iterations to perform erosion

    :return
    erode_img: numpy array representing eroded image
    '''
    kernel = np.ones((kernel_size_x, kernel_size_y), 'uint8') # does using other default values help?
    erode_img = cv2.erode(image, kernel, iterations)

    return erode_img

def convert_gray2red(image):
    '''
    Function: converts a gray image into a red image(uint8) and saved it as image_red.png
    :return:
    '''
    zeros = np.zeros((1024, 1024), dtype=np.uint8)
    img_gray = image.astype(np.uint8)
    print("img_gray dtype:", img_gray.dtype)
    color_image = np.stack((img_gray, zeros, zeros), axis=2)
    # print("ColorImage shape:", color_image.shape)
    # print("ColorImage dtype:", color_image.dtype)
    skimage.io.imsave('image_red.png', color_image)
    print("Image saved as image_red.png")

    return color_image


def save_img(fname, img):

    '''
Function: Saves an image into a specified file type
Parameters
fname: str, Target filename. Filename must include image format such as .tiff, .png, .jpg
img: image to be saved. Typically a numpy.ndarray

'''
    cv2.imwrite(fname, img)
    print('Saving completed')

    return None

def img_crop(img,dim):
    '''
    Function: Crops image by performing array indexing. Specify coordinates for cropping
    Inputs:
    img:  image to be cropped
    coordinates: int values, Start y: End y, Start x: End x
                Cropped Image will have pixel from Start y to (End y) -1 and Start x to (End x) -1
    dim: tuple, (Start y, End y, Start x, End x)
    Start y: The starting y-coordinate.
    End y: The ending y-coordinate.
    Start x: The starting x-coordinate of the slice.
    End x: The ending x-axis coordinate of the slice.
    '''
    # coordinates=[start y:end y, start x: end x]

    cropped = img[dim[0]:dim[1],dim[2]: dim[3]]
    # not working, cropped is empty array
    print(cropped)
    print('Cropping Completed')

    return cropped

#Super-Resolution
# bilinear_img = cv2.resize(img_equalizeHist_suppressed_equalizehist_multiplied,None, fx = 1.2, fy = 1.2, interpolation = cv2.INTER_CUBIC)
# #fx and fy is the factor by which height and width is increased
# #try interpolation = cv2.INTER_NEAREST,cv2.INTER_CUBIC
# cv2.imshow('bilinear_interpol',bilinear_img)
# cv2.waitKey(0)

###-----------------------------------------------------------------------------------------------------------------------###
###                                                 Body                                                                  ###
###                                                                                                                       ###
###-----------------------------------------------------------------------------------------------------------------------###


# load tif
filename='movie.tif'
load(filename)

#selecting a specific slice
img = cv2.imread('stack0,channel0.png')

# print(type(img))
# print('Dimension of image',img.shape)
# print('Data type of image(original)',img.dtype)
# cv2.imshow documentation 'https://docs.opencv.org/4.x/d7/dfc/group__highgui.html#ga453d42fe4cb60e5723281a89973ee563'
cv2.imshow('image_original',img)
cv2.waitKey(0)

#modify contrast and brightness
img_mod_con_bri= mod_con_bri(2.0,0,1,img)
cv2.imshow('image_modified',img_mod_con_bri)
cv2.waitKey(0)

# modify saturation
# img_mod_sat=mod_saturation(img_mod_con_bri,1.0)
# cv2.imshow('saturation',img_mod_sat)
# cv2.waitKey(0)

#sharpen
sharpened_image=sharpen(img_mod_con_bri)
cv2.imshow('image_sharpened',sharpened_image)
cv2.waitKey(0)

#erosion
img_eroded=erosion(img_mod_con_bri,2,2,2)
cv2.imshow('image_eroded',img_eroded)
cv2.waitKey(0)
print('Eroded image Datatype',img_eroded.dtype)
# save_img('img_eroded_s0c0.png',img_eroded)

# crop=img_eroded[200:700,350:950]
# save_img('ISRtest.png',crop)

#skimage.io.imsave('Best.png',img_eroded)


# #Gray Scaling
#converts from RGB to gray
img_gray=convert_grayscale(img_eroded)
# cv2.imshow('gray_image',img_gray)

#save a gray scale image as a red image
#convert_gray2red(img_gray)

# # transposing data for specific CellPose need
# resized_data = np.transpose(data, (0, 3, 1, 2))
# print(resized_data.shape)


# plt.hist(img_gray.ravel(),256,[0,256], label='Gray Image'); #img.ravel() flattens array into a 1D array without changing input type
# plt.legend(prop={'size': 15})
# plt.savefig('pg6_gray.jpg')

# # Histogram Equalization
# Documentation cv2.equalizeHist https://docs.opencv.org/4.x/d5/daf/tutorial_py_histogram_equalization.html #
# # 4-step Process                                                                                    #
# # 1.Equalize Histogram                                                                              #
# # 2.Suppress values                                                                                 #
# # 3.Equalize Histogram                                                                              #
# # 4. Mutiply intensity by a factor to increase contrast                                             #
# # --------------------------------------------------------------------------------------------------#

#1.Equalize Histogram
## takes 8 bit single channel image, outputs 8 bit image
img_equalizeHist = cv2.equalizeHist(img_gray)
cv2.imshow('image_original_equalizehist',img_equalizeHist)

#visualizing data using histogram
plt.hist(img_equalizeHist.ravel(),256,[0,256],label='HistogramEqualized');
plt.legend(prop={'size': 15})
plt.savefig('stack0,channel0_equalizehist.jpg')

# 2.Suppress values
img_equalizeHist[(210>img_equalizeHist)] = 0.0
img_equalizeHist_suppressed=img_equalizeHist
cv2.imshow('image_original_equalizehist_suppressed',img_equalizeHist_suppressed)
cv2.waitKey(0)

plt.hist(img_equalizeHist_suppressed.ravel(),256,[0,256],label='HistoEqualized_Suppress');
plt.legend(prop={'size': 15})
plt.savefig('stack0,channel0_equalizehist_suppressed.jpg')
print('Image Datatype',img_equalizeHist_suppressed.dtype)

# 3.Equalize Histogram
# NOTE: Image becomes very saturated
img_equalizeHist_suppressed_equalizeHist=cv2.equalizeHist(img_equalizeHist_suppressed)

cv2.imshow('image_original_equalizehist_suppressed_equalizehist',img_equalizeHist_suppressed_equalizeHist)
cv2.waitKey(0)
# plt.hist(img_equalizeHist_suppressed_equalizeHist.ravel(),label='HistoEqualized_Suppress_HistEqualized')
# plt.legend(prop={'size': 15})
# plt.savefig('stack0,channel0_equalizehist_suppressed_equalizedhist.jpg')
img_equalizeHist_suppressed_equalizeHist=mod_con_bri(1,-30,1,img_equalizeHist_suppressed_equalizeHist)
cv2.imshow('Reduce brightness',img_equalizeHist_suppressed_equalizeHist)
cv2.waitKey(0)
mod_img=convert_gray2red(img_equalizeHist_suppressed_equalizeHist)

