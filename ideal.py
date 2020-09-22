# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2

import numpy as np
import math
import os
from objloader_simple import *
from calibration import *

# load the two input images
dir_name = os.getcwd()
imageA = cv2.imread(os.path.join(dir_name, 'backsnap/empty.jpg'))
imageB = frame
# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)



# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
#(score, diff) = compare_ssim(imageA, imageB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))


#chanel0
(score0, diff0) = compare_ssim(imageA[:,:,0], imageB[:,:,0], full=True)
#(score, diff) = compare_ssim(imageA, imageB, full=True)
diff0 = (diff0 * 255).astype("uint8")
print("SSIM0: {}".format(score0))

#chanel1
(score1, diff1) = compare_ssim(imageA[:,:,1], imageB[:,:,1], full=True)
#(score, diff) = compare_ssim(imageA, imageB, full=True)
diff1 = (diff1 * 255).astype("uint8")
print("SSIM1: {}".format(score1))

#chanel2
(score2, diff2) = compare_ssim(imageA[:,:,2], imageB[:,:,2], full=True)
#(score, diff) = compare_ssim(imageA, imageB, full=True)
diff2 = (diff2 * 255).astype("uint8")
print("SSIM2: {}".format(score2))


dst1 = cv2.addWeighted(diff0,0.333,diff1,0.333,0)
dst = cv2.addWeighted(dst1,1,diff2,0.333,0)

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
#thresh = cv2.threshold(diff, 10, 255,
#   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

thresh = cv2.threshold(dst, 10, 255,
   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#thresh0 = cv2.threshold(diff0, 10, 255,
#    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#thresh1 = cv2.threshold(diff1, 10, 255,
#    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#thresh2 = cv2.threshold(diff2, 10, 255,
#    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#dst1 = cv2.addWeighted(thresh0,0.333,thresh1,0.333,0)
#dst = cv2.addWeighted(dst1,0.333,thresh2,0.333,0)

# contours
#cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#	cv2.CHAIN_APPROX_SIMPLE)
#cnts = imutils.grab_contours(cnts)

# to close holes need masks with dilation
kernal = np.ones((3,3), np.uint8)
diletion = cv2.dilate(thresh, kernal, iterations=3)
#erosia
erosion = cv2.erode(diletion, kernal, iterations=3)


cv2.namedWindow('erosion', cv2.WINDOW_NORMAL)
cv2.imshow('erosion', erosion)

cv2.namedWindow('diletion', cv2.WINDOW_NORMAL)
cv2.imshow('diletion', diletion)

cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
cv2.imshow('thresh', thresh)

cv2.waitKey()
cv2.destroyAllWindows()