import numpy as np
from PIL import Image, ImageSequence
import matplotlib.pyplot as plt
import cv2
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


# Load image - give option to run code again or to keep going adding code to same directory
movie = Image.open('/Users/dannycowen/PycharmProjects/3dvisualisation/movie.tif')

#fix the colour problem (not fixed yet)
palette = movie.palette.colors
palette = palette.keys()
palette = [list(elem) for elem in palette]
movie_cmap = colors.ListedColormap(palette, name='movie_colourmap')


# create directory. directories of slices of each image inputted will be stored here
# need to change this filepath depending on the user

if os.path.isdir("/Users/dannycowen/PycharmProjects/3dvisualisation/moviecode/movieframes"):
    # if folder exists, change into folder
    os.chdir("/Users/dannycowen/PycharmProjects/3dvisualisation/moviecode/movieframes")
    print("parent directory movieframes found")
else:
    # if folder doesn't exist, create folder and change into it
    os.mkdir("/Users/dannycowen/PycharmProjects/3dvisualisation/moviecode/movieframes")
    os.chdir("/Users/dannycowen/PycharmProjects/3dvisualisation/moviecode/movieframes")
    print("new parent directory 'movieframes' created")


answer = input("Would you like to load a new movie, or continue working on one? (new/continue)")

path = ""

if answer == 'new':
    try:
        # HERE CAN GIVE OPTION TO NAME DIRECTORY AS PREFERRED
        directory = input('enter a name for your new directory containing which will contain each slice as an image')
        parent_dir = "/Users/dannycowen/PycharmProjects/3dvisualisation/moviecode/movieframes"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
        os.chdir(path)
        print("Directory '% s' created" % directory)
        print("Currently operating in %s" % os.getcwd())
    except FileExistsError:
        print('directory name already in use. enter another.')
        directory = input('enter the name of the directory you wish to reenter')
        parent_dir = "/Users/dannycowen/PycharmProjects/3dvisualisation/moviecode/movieframes"
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)
        os.chdir(path)
        print("Directory '% s' created" % directory)
        print("Currently operating in %s" %os.getcwd())
elif answer != 'continue':
    print("Make sure you input one of the available options")
elif answer == 'continue':
    # if user wants to reenter into a directory of slices already created
    parent_dir = "/Users/dannycowen/PycharmProjects/3dvisualisation/moviecode/movieframes"
    directory = input("enter the name of the directory containing the slices")
    path = os.path.join(parent_dir, directory)
    os.chdir(path)
    print('Currently operating in %s' % os.getcwd())

# save no. of frames in the movie
no_frames = movie.n_frames

# create an image for each frame in the movie, save to current working directory
# check if images have already been created to save time

ans = input('have the separate slices of the movie been created as tif files already?'
            'if so, they can be found in this directory. please answer yes/no')

if ans == 'no':
    for frame in range(0, no_frames):
        movie.seek(frame)
        x = path + '/slice' + str(frame+1) + '.tif'
        movie.save(x)
elif ans != 'yes' and ans != 'no':
    print('check your input - you did not answer yes or no')

# establish how many channels and z slices are in the movie

z_slices = int(input("How many z slices per time point in the loaded image?"))

# COULD do a statement here asking are you sure

channels = int(input('how many channels in the image?'))

# COULD do a statement here asking are you sure

# separate red channel into its own array
z_across_t_red = []
for i in range(0, no_frames, 10):
    cells_z = []
    for j in range(1, (z_slices*channels)+1, channels): # create list of red channels
        temp_im = Image.open('slice%s.tif' % (j+i))
        imdata = temp_im.getdata()
        x = np.reshape(imdata, (temp_im.height, temp_im.width))
        x = np.asarray(x, dtype=np.uint8)
        cells_z.append(x)
    z_across_t_red.append(cells_z)

# separate blue channel into its own array

z_across_t_blue = []
for i in range(0, no_frames, 10):
    cells_z = []
    for j in range(2, (z_slices*channels)+1, channels): # create list of blue channels
        temp_im = Image.open('slice%s.tif' % (j+i))
        imdata = temp_im.getdata()
        x = np.reshape(imdata, (temp_im.height, temp_im.width))
        x = np.asarray(x, dtype=np.uint8)
        cells_z.append(x)
    z_across_t_blue.append(cells_z)

# rearrange array shapes

red_images_array = np.asarray(z_across_t_red)
blue_images_array = np.asarray(z_across_t_blue)

red_images_array = np.moveaxis(red_images_array, 2, 0)
red_images_array = np.moveaxis(red_images_array, 3, 1)
blue_images_array = np.moveaxis(blue_images_array, 2, 0)
blue_images_array = np.moveaxis(blue_images_array, 3, 1)

combined_images_array = red_images_array + blue_images_array

# !!!ASK USER WHICH CHANNELS THEY WANT TO VIEW

# Fixing random state for reproducibility
np.random.seed(19680801)

# will have to ask user which channel they want to view before running this code
# scroller class for 3d movies

class IndexTrackerMovie:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use arrow keys. up/down = scroll z-stack, left/right = scroll time series')

        self.X = X
        rows, cols, self.tslices, self.zslices = X.shape
        self.zind = self.zslices // 2
        self.tind = self.tslices // 2

        # This next line shows the indexing required to scroll through space, ie (x, y, z)
        self.im = ax.imshow(self.X[:, :, self.tind, self.zind])
        self.update()

    def on_key_press(self, event):
        print("%s" % (event.key))
        if event.key == 'up':
            self.zind = (self.zind + 1) % self.zslices
        if event.key == 'down':
            self.zind = (self.zind - 1) % self.zslices
        if event.key == 'left':
            self.tind = (self.tind - 1) % self.tslices
        if event.key == 'right':
            self.tind = (self.tind + 1) % self.tslices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.tind, self.zind])
        self.ax.set_ylabel('z-slice %s, t-slice %s' % (self.zind, self.tind))
        self.im.axes.figure.canvas.draw()

# Scroller class for 3d Stacks

class IndexTrackerStack:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use arrow keys. up/down = scroll z-stack, left/right = scroll time series')

        self.X = X
        rows, cols, self.zslices = X.shape
        self.zind = self.zslices // 2

        # This next line shows the indexing required to scroll through space, ie (x, y, z)
        self.im = ax.imshow(self.X[:, :, self.zind])
        self.update()

    def on_key_press(self, event):
        print("%s" % (event.key))
        if event.key == 'up':
            self.zind = (self.zind + 1) % self.zslices
        if event.key == 'down':
            self.zind = (self.zind - 1) % self.zslices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.zind])
        self.ax.set_ylabel('z-slice %s' % (self.zind))
        self.im.axes.figure.canvas.draw()

fig, ax = plt.subplots(1, 1)

# reshape array to be able to scroll through images


tracker = IndexTrackerMovie(ax, red_images_array)

fig.canvas.mpl_connect('key_press_event', tracker.on_key_press)
plt.show()
