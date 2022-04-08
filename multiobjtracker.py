from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import pandas as pd
import csv

#pip install opencv-contrib-python==4.5.5.64

#TRACKING CODE to track CellPose's masks

# opening the csv file in 'w+' mode
file = open('Coordinates.csv' , 'w+' , newline='')
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, default="do3Dmovie.mp4",
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

# initialize a dictionary that maps strings to their corresponding
# OpenCV object tracker implementations
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.legacy.TrackerCSRT_create,
	"kcf": cv2.legacy.TrackerKCF_create,
	"boosting": cv2.legacy.TrackerBoosting_create,
	"mil": cv2.legacy.TrackerMIL_create,
	"tld": cv2.legacy.TrackerTLD_create,
	"medianflow": cv2.legacy.TrackerMedianFlow_create,
	"mosse": cv2.legacy.TrackerMOSSE_create
}
# initialize OpenCV's special multi-object tracker
trackers = cv2.legacy.MultiTracker_create()

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame
	# check to see if we have reached the end of the stream
	if frame is None:
		break
	# resize the frame (so we can process it faster)
	frame = imutils.resize(frame, width=600)

	# grab the updated bounding box coordinates (if any) for each
	# object that is being tracked
	(success, boxes) = trackers.update(frame)

	ls = [] #empty list to store tuple of cords for each box/cell

	# loop over the bounding boxes and draw then on the frame
	cellcounter = 0
	for box in boxes :
		cellcounter=cellcounter+1
		(x , y , w , h) = [int(v) for v in box]
		cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 255 , 0) , 2)
		# put (x,y) into a tuple and save in a list
		cord=(x,y)
		ls.append(cellcounter)
		ls.append(cord)
		write = csv.writer(file)
		write.writerow(ls)
	#after processing whole video file
	# write = csv.writer(file)
	# sen="Total no of cell tracked:" + str(cellcounter)
	# write.writerow(sen)


		# cv2.putText(frame , "Tracking" , (50 , 80) , cv2.FONT_HERSHEY_COMPLEX , 0.7 , (0 , 255 , 0) , 1)
		# # # show x,y coordinate:
		# cv2.putText(frame , "X =" , (0 , 30) , cv2.FONT_HERSHEY_COMPLEX , 0.7 , (0 , 255 , 0) , 1)
		# cv2.putText(frame , str(int(x)) , (40 , 30) , cv2.FONT_HERSHEY_COMPLEX , 0.7 , (0 , 255 , 0) , 1)
		# cv2.putText(frame , "Y =" , (100 , 30) , cv2.FONT_HERSHEY_COMPLEX , 0.7 , (0 , 255 , 0) , 1)
		# cv2.putText(frame , str(int(y)) , (140 , 30) , cv2.FONT_HERSHEY_COMPLEX , 0.7 , (0 , 255 , 0) , 1)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the 's' key is selected, we are going to "select" a bounding
	# box to track
	if key == ord("s"):
		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		ls_box = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)
		# create a new object tracker for the bounding box and add it
		# to our multi-object tracker
		tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
		trackers.add(tracker, frame, ls_box)

	# if the `q` key was pressed, break from the loop
	elif key == ord("q"):
		break
# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()
# otherwise, release the file pointer
else:
	vs.release()
# close all windows
cv2.destroyAllWindows()
