# converting video to images
import cv2; print(cv2.__version__)



'''----------------------
Converting first video to frames
--------------------------'''

vidcap = cv2.VideoCapture('/home/tharunshiv/fyp/spot-the-diff/data/video/videoOriginal.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("/home/tharunshiv/fyp/spot-the-diff/data/frames/"+str(sec)+".jpg", image)	
    return hasFrames

sec = 0
frameRate = 1 # it will capture image in each 0.5 second
success = getFrame(sec)

while success:
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

print("Converted First video into Frames")
print("We got ", sec, " frames")

'''----------------------
Converting second video to frames
--------------------------'''
vidcap = cv2.VideoCapture('/home/tharunshiv/fyp/spot-the-diff/data/video/videoLabelled.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite("/home/tharunshiv/fyp/spot-the-diff/data/frames/"+str(sec)+"_2.jpg", image)     # save frame as JPG file
    return hasFrames
sec = 0
frameRate = 1 # it will capture image in each 0.5 second
success = getFrame(sec)
while success:
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

print("\nConverted Second video into Frames")
print("We got ", sec, " frames\n")

print("Now finding the difference thresholding between them\n")

'''----------------------
Finding the Diff
--------------------------'''
# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2

for i in range(sec):
	# load the two input images
	frameBasePath = '/home/tharunshiv/fyp/spot-the-diff/data/frames/'
	imageA = cv2.imread(frameBasePath+str(i)+'.jpg')
	imageB = cv2.imread(frameBasePath+str(i)+'_2.jpg')

	print(frameBasePath+str(i)+'.jpg')
	print(frameBasePath+str(i)+'_2.jpg')


	# convert the images to grayscale
	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

	# compute the Structural Similarity Index (SSIM) between the two
	# images, ensuring that the difference image is returned
	(score, diff) = compare_ssim(grayA, grayB, full=True)
	diff = (diff * 255).astype("uint8")
	# print("SSIM: {}".format(score))

	# threshold the difference image, followed by finding contours to
	# obtain the regions of the two input images that differ
	thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	

	cv2.imwrite("/home/tharunshiv/fyp/spot-the-diff/data/frames/"+str(i)+"_diff.jpg", thresh) 


	# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	# 	cv2.CHAIN_APPROX_SIMPLE)
	# cnts = imutils.grab_contours(cnts)

	# # loop over the contours
	# for c in cnts:
	# 	# compute the bounding box of the contour and then draw the
	# 	# bounding box on both input images to represent where the two
	# 	# images differ
	# 	(x, y, w, h) = cv2.boundingRect(c)
	# 	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	# 	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# show the output images
	# cv2.imshow("Original", imageA)
	# cv2.imshow("Modified", imageB)
	# cv2.imshow("Diff", diff)
	# cv2.imshow("Thresh", thresh)
	# cv2.waitKey(0)

print("Found the thresholded diff between then and saved")