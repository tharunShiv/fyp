# # converting video to images
# import cv2; print(cv2.__version__)
# import numpy as np

# '''----------------------
# Converting first video to frames
# --------------------------'''

# vidcap = cv2.VideoCapture('/home/tharunshiv/image-difference/virus.mp4')
# def getFrame(sec, count):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#     hasFrames,image = vidcap.read()
#     if hasFrames:
#         cv2.imwrite("/home/tharunshiv/image-difference/one/"+str(count)+".jpg", image)	
#     return hasFrames

# count = 0
# sec = 0
# frameRate = 0.05 # it will capture image in each 0.5 second
# success = getFrame(sec, count)

# while success:
# 	count = count + 1
# 	sec = sec + frameRate
# 	sec = round(sec, 2)
# 	success = getFrame(sec, count)

# print("Converted First video into Frames")
# print("We got ", count, " frames")

# '''----------------------
# Converting second video to frames
# --------------------------'''
# vidcap = cv2.VideoCapture('/home/tharunshiv/image-difference/virusMod.mp4')
# def getFrame(sec, count):
#     vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
#     hasFrames,image = vidcap.read()
#     if hasFrames:
#         cv2.imwrite("/home/tharunshiv/image-difference/two/"+str(count)+"_2.jpg", image)     # save frame as JPG file
#     return hasFrames

# count = 0
# sec = 0
# frameRate = 0.05 # it will capture image in each 0.5 second
# success = getFrame(sec, count)

# while success:
# 	count = count+1
# 	sec = sec+frameRate
# 	sec = round(sec, 2)
# 	success = getFrame(sec, count)

# count = int(count)
# print("\nConverted Second video into Frames")
# print("We got ", count, " frames\n")

# print("Now finding the difference thresholding between them\n")

# '''----------------------
# Finding the Diff
# --------------------------'''
# # import the necessary packages
# from skimage.measure import compare_ssim
# import argparse
# import imutils
# import cv2

# for i in range(count):
# 	# load the two input images
# 	frameBasePath = '/home/tharunshiv/image-difference/one/'
# 	imageA = cv2.imread(frameBasePath+str(i)+'.jpg')
# 	frameBasePath2 = '/home/tharunshiv/image-difference/two/'
# 	imageB = cv2.imread(frameBasePath2+str(i)+'_2.jpg')

# 	print(frameBasePath+str(i)+'.jpg')
# 	print(frameBasePath2+str(i)+'_2.jpg')
# 	# convert the images to grayscale
# 	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
# 	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# 	# kernel = np.ones((5, 5), np.float32)/32
# 	# grayA = cv2.filter2D(grayA, -1, kernel)

# 	# kernel = np.ones((5, 5), np.float32)/32
# 	# grayB = cv2.filter2D(grayB, -1, kernel)

# 	grayA = cv2.bilateralFilter(grayA,9,75,75)
# 	grayB = cv2.bilateralFilter(grayB,9,75,75)
	

# 	# compute the Structural Similarity Index (SSIM) between the two
# 	# images, ensuring that the difference image is returned
# 	(score, diff) = compare_ssim(grayA, grayB, full=True)
# 	diff = (diff * 255).astype("uint8")
# 	# print("SSIM: {}".format(score))

# 	# threshold the difference image, followed by finding contours to
# 	# obtain the regions of the two input images that differ
# 	thresh = cv2.threshold(diff, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	

# 	cv2.imwrite("/home/tharunshiv/image-difference/three/"+str(i)+"_diff.jpg", thresh) 


# 	# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 	# 	cv2.CHAIN_APPROX_SIMPLE)
# 	# cnts = imutils.grab_contours(cnts)

# 	# # loop over the contours
# 	# for c in cnts:
# 	# 	# compute the bounding box of the contour and then draw the
# 	# 	# bounding box on both input images to represent where the two
# 	# 	# images differ
# 	# 	(x, y, w, h) = cv2.boundingRect(c)
# 	# 	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
# 	# 	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 	# show the output images
# 	# cv2.imshow("Original", imageA)
# 	# cv2.imshow("Modified", imageB)
# 	# cv2.imshow("Diff", diff)
# 	# cv2.imshow("Thresh", thresh)
# 	# cv2.waitKey(0)

# print("Found the thresholded diff between then and saved")

# # USAGE
# # python image_diff.py --first images/original_01.png --second images/modified_01.png

# # import the necessary packages
# from skimage.measure import compare_ssim
# import argparse
# import imutils
# import cv2

# import numpy as np

# # # construct the argument parse and parse the arguments
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-f", "--first", required=True,
# # 	help="first input image")
# # ap.add_argument("-s", "--second", required=True,
# # 	help="second")
# # args = vars(ap.parse_args())

# # # load the two input images
# # imageA = cv2.imread(args["first"])
# # imageB = cv2.imread(args["second"])


# for i in range(count):
# 	frameBasePath = '/home/tharunshiv/image-difference/one/'
# 	imageA = cv2.imread(frameBasePath+str(i)+'.jpg')
# 	frameBasePath2 = '/home/tharunshiv/image-difference/two/'
# 	imageB = cv2.imread(frameBasePath2+str(i)+'_2.jpg')

# 	# convert the images to grayscale
# 	grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
# 	grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# 	grayA = cv2.bilateralFilter(grayA,9,75,75)
# 	grayB = cv2.bilateralFilter(grayB,9,75,75)

# 	# cv2.imshow("grayB", grayB)
# 	# cv2.imshow("im", grayB)
# 	# cv2.waitKey(0)

# 	# compute the Structural Similarity Index (SSIM) between the two
# 	# images, ensuring that the difference image is returned
# 	(score, diff) = compare_ssim(grayA, grayB, full=True)
# 	diff = (diff * 255).astype("uint8")
# 	print("SSIM: {}".format(score))

# 	# threshold the difference image, followed by finding contours to
# 	# obtain the regions of the two input images that differ
# 	thresh = cv2.threshold(diff, 0, 255,
# 		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# 	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
# 		cv2.CHAIN_APPROX_SIMPLE)
# 	cnts = imutils.grab_contours(cnts)

# 	# loop over the contours
# 	for c in cnts:
# 		# compute the bounding box of the contour and then draw the
# 		# bounding box on both input images to represent where the two
# 		# images differ
# 		(x, y, w, h) = cv2.boundingRect(c)
# 		cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
# 		cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
	
# 	cv2.imwrite("/home/tharunshiv/image-difference/one/"+str(i)+".jpg", imageA)     # save frame as JPG file
# 	cv2.imwrite("/home/tharunshiv/image-difference/two/"+str(i)+"_2.jpg", imageB)     # save frame as JPG file

# 	# show the output images
# 	# cv2.imshow("Original", imageA)
# 	# cv2.imshow("Modified", imageB)
# 	# cv2.imshow("Diff", diff)
# 	# cv2.imshow("Thresh", thresh)
# 	# cv2.waitKey(0)

# #######


import cv2
import numpy as np
import os
 
from os.path import isfile, join
 
def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    files.sort()
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,0x7634706d, fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

def main():
	pathIn = '/home/tharunshiv/image-difference/one/'
	pathOut = '/home/tharunshiv/image-difference/videoOut1.mp4'
	fps = 10.0
	convert_frames_to_video(pathIn, pathOut, fps)

	pathIn = '/home/tharunshiv/image-difference/two/'
	pathOut = '/home/tharunshiv/image-difference/videoOut2.mp4'
	fps = 10.0
	convert_frames_to_video(pathIn, pathOut, fps)

 
if __name__=="__main__":
    main()