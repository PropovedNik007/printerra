# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils

import cv2
import numpy as np
import math
import os
from objloader_simple import *


def main():
    # octoprint 1296x972
    # 2592x1944
    # matrix of camera parameters
    # fx 0  cx
    # 0  fy cy
    # 0  0   1
    # fx = focal length/
    # fy
    # cx
    # cy
    # ObjectPoints = []
    # ImagePoints = []
    distCoeffs = []
    rvec = np.float32([])
    tvec = np.float32([])
    #camera matrix for calibration
    a = np.array([[2407.14, 0, 1296], [0, 2407.14, 972], [0, 0, 1]])
    # coordinates from 3D points
    # test fox and box
    #objectPoints = np.array([[0, 0, 0], [205, 0, 0],[205, 0, 70], [0, 0, 70], [0, 155, 0], [0, 155, 70]])
    # XYZ calibration model
    objectPoints = np.array([
        [-100, -100, 20],
        [-50, -100, 30],
        [0, -100, 40],
        [50, -100, 50],

        [-100, -50, 20],
        [-50, -50, 30],
        [0, -50, 40],
        [50, -50, 50],

        [-100, -50, 50],
        [-50, -50, 70],
        [0, -50, 90],
        [50, -50, 110],
        [100, -50, 110],

        [-100, 0, 50],
        [-100, 0, 70],
        [50, 0, 110],
        [50, 0, 150],
        [100, 0, 150],

        [-100, 50, 70],

        [-100, 50, 150],
        [-50, 50, 150],
        [50, 100, 200],
        [50, 50, 200],
        [100, 50, 200]])
    # izmerennie znacheniya v Paint. nuzhno *2 v realnie razmeri
    # 2D points calibration fox and box
    #ImagePoints = np.array([[722, 908], [1257, 868], [1235, 598], [720, 445], [220, 861], [249, 557])
    # real size pixel coordinates
    #imagePoints = 2 * np.array([[722, 908], [1257, 868], [1235, 598], [720, 445], [220, 861], [249, 557]])
    # calibration model 2D points
    imagePoints = 2 * np.array([
        [783, 704],
        [981, 649],
        [1119, 609],
        [1222, 579],

        [544, 699],
        [749, 652],
        [898, 618],
        [1012, 589],

        [547, 515],
        [746, 450],
        [890, 394],
        [1004, 354],
        [1097, 392],

        [380, 542],
        [385, 440],
        [845, 385],
        [842, 252],
        [939, 293],

        [258, 469],

        [283, 135],
        [456, 197],
        [611, 187],
        [716, 145],
        [810, 189]])

    result = cv2.solvePnP(np.float32(objectPoints), np.float32(imagePoints), np.float32(a), np.float32(distCoeffs))

    status = result[0]
    rvec = result[1]
    tvec = result[2]
    projectedPoints = cv2.projectPoints(np.float32(objectPoints), rvec, tvec, np.float32(a), np.float32(distCoeffs), np.float32(imagePoints))
    points = projectedPoints[0]


    #print(status)
    #print(rvec)
    #print(tvec)
    #print(imagePoints)
    #print(points)

    myObjPoint = np.array([102, 0, 35])
    myProjectedPoint = cv2.projectPoints(np.float32(myObjPoint), rvec, tvec, np.float32(a), np.float32(distCoeffs))
    myPrPoint = 0.5 * myProjectedPoint[0]
    #print(myPrPoint)

    dir_name = os.getcwd()
    #frame = cv2.imread(os.path.join(dir_name, 'snapshots/123.jpg'))
    frame = cv2.imread(os.path.join(dir_name, 'backsnap/empty.jpg'))
    # Load 3D model from OBJ file
    #obj = OBJ(os.path.join(dir_name, 'models/fox.obj'), swapyz=True)
    #obj = OBJ(os.path.join(dir_name, 'models/calibration.obj'), swapyz=False)
    obj = OBJ(os.path.join(dir_name, 'models/Baby_Yoda.obj'), swapyz=True)
    verticesx2 = cv2.projectPoints(np.float32(obj.vertices), rvec, tvec, np.float32(a), np.float32(distCoeffs))
    vertices = 0.5 * verticesx2[0]


#vertices points
    #for i, val in enumerate(vertices):
    #    #print (i, ",", val[0][0])
    #    cv2.circle(frame, (int(val[0][0]), int(val[0][1])), 0.5, (0, 0, 255), -1)

    #cv2.circle(frame,(720, 913), 40, (0,0,255), -1)
    #cv2.circle(frame,(720, 913), 40, (0,0,255), -1)
    #cv2.circle(frame,(int(myPrPoint[0][0][0]), int(myPrPoint[0][0][1])), 40, (0,0,255), -1)
    frame = render(obj, vertices, frame, color=False)

    frame1 = cv2.imread(os.path.join(dir_name, 'backsnap/nonempty.jpg'))
    cv2.namedWindow('reference', cv2.WINDOW_NORMAL)
    cv2.imshow('reference', frame1)

    cv2.namedWindow('ProjectedModel', cv2.WINDOW_NORMAL)
    cv2.imshow('ProjectedModel', frame)
    #ideal(frame)
    #real()
    #sravnenie
    #kriteriy PASCAL VOC: S1/S2 < 0.5
    #peresechenie A and B
    s1 = cv2.countNonZero(cv2.bitwise_and(real(), ideal(frame)))
    print("S1:", s1)
    #ob'edenenie S1 and S2
    s2 = cv2.countNonZero(cv2.bitwise_or(real(), ideal(frame)))
    print("S2:", s2)
    pascal_voc = s1 / s2
    if pascal_voc < 0.5:
        print(pascal_voc, "WARNING")
    else:
        print(pascal_voc, "OK")


    cv2.waitKey()
    cv2.destroyAllWindows()
    return 0




def render(obj, vertices, frame, color=False):
    """
    Render a loaded obj model into the current video frame
    """
    #vertices = obj.vertices
    #scale_matrix = np.eye(3) * 3
    #h, w = model.shape
    for face in obj.faces:
        face_vertices = face[0]
        points = np.array([vertices[vertex - 1] for vertex in face_vertices])
        points = np.array([p[0] for p in points])
        #points = np.dot(points, scale_matrix)
        # render model in the middle of the reference surface. To do so,
        # model points must be displaced
        imgpts = np.int32(points)
        if color is False:
            cv2.fillConvexPoly(frame, imgpts, (11, 252, 3))
        else:
            color = hex_to_rgb(face[-1])
            color = color[::-1]  # reverse
            cv2.fillConvexPoly(frame, imgpts, color)
    return frame


def hex_to_rgb(hex_color):
    """
    Helper function to convert hex strings to RGB
    """
    #hex_color = hex_color.lstrip('#0bfc03')
    hex_color = hex_color.lstrip("#0bfc03")
    h_len = len(hex_color)
    return tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))




def ideal(frame):
    # load the two input images
    dir_name = os.getcwd()
    imageA = cv2.imread(os.path.join(dir_name, 'backsnap/empty.jpg'))
    idealImageB = frame
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    idealGrayB = cv2.cvtColor(idealImageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (idealScore, idealdiff) = compare_ssim(grayA, idealGrayB, full=True)
    #(score, diff) = compare_ssim(imageA, idealImageB, full=True)
    idealdiff = (idealdiff * 255).astype("uint8")
    #print("SSIM: {}".format(score))

    #chanel0
    (idealScore0, idealDiff0) = compare_ssim(imageA[:,:,0], idealImageB[:,:,0], full=True)
    #(score, diff) = compare_ssim(imageA, imageB, full=True)
    idealDiff0 = (idealDiff0 * 255).astype("uint8")
    #print("SSIM0: {}".format(idealScore0))

    #chanel1
    (idealScore1, idealDiff1) = compare_ssim(imageA[:,:,1], idealImageB[:,:,1], full=True)
    #(score, diff) = compare_ssim(imageA, imageB, full=True)
    idealDiff1 = (idealDiff1 * 255).astype("uint8")
    #print("SSIM1: {}".format(idealScore1))

    #chanel2
    (idealScore2, idealDiff2) = compare_ssim(imageA[:,:,2], idealImageB[:,:,2], full=True)
    #(score, diff) = compare_ssim(imageA, imageB, full=True)
    idealDiff2 = (idealDiff2 * 255).astype("uint8")
    #print("SSIM2: {}".format(idealScore2))

    idealDst1 = cv2.addWeighted(idealDiff0,0.333,idealDiff1,0.333,0)
    idealDst = cv2.addWeighted(idealDst1,1,idealDiff2,0.333,0)

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    #thresh = cv2.threshold(diff, 10, 255,
    #   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    idealThresh = cv2.threshold(idealDst, 10, 255,
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
    idealKernal = np.ones((3,3), np.uint8)
    idealDiletion = cv2.dilate(idealThresh, idealKernal, iterations=3)
    #erosia
    idealErosion = cv2.erode(idealDiletion, idealKernal, iterations=3)


    cv2.namedWindow('idealErosion', cv2.WINDOW_NORMAL)
    cv2.imshow('idealErosion', idealErosion)

    #cv2.namedWindow('diletion', cv2.WINDOW_NORMAL)
    #cv2.imshow('diletion', diletion)

    #cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    #cv2.imshow('thresh', thresh)
    return idealErosion

def real():
    # load the two input images
    dir_name = os.getcwd()
    imageA = cv2.imread(os.path.join(dir_name, 'backsnap/empty.jpg'))
    realImageB = cv2.imread(os.path.join(dir_name, 'backsnap/nonempty.jpg'))
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    realGrayB = cv2.cvtColor(realImageB, cv2.COLOR_BGR2GRAY)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (realScore, realDiff) = compare_ssim(grayA, realGrayB, full=True)
    # (score, diff) = compare_ssim(imageA, imageB, full=True)
    realDiff = (realDiff * 255).astype("uint8")
    #print("SSIM: {}".format(realScore))

    # color chanel0
    (realScore0, realDiff0) = compare_ssim(imageA[:, :, 0], realImageB[:, :, 0], full=True)
    # (score, diff) = compare_ssim(imageA, imageB, full=True)
    realDiff0 = (realDiff0 * 255).astype("uint8")
    #print("SSIM0: {}".format(realScore0))

    # color chanel1
    (realScore1, realDiff1) = compare_ssim(imageA[:, :, 1], realImageB[:, :, 1], full=True)
    # (score, diff) = compare_ssim(imageA, imageB, full=True)
    realDiff1 = (realDiff1 * 255).astype("uint8")
    #print("SSIM1: {}".format(realScore1))

    # color chanel2
    (realScore2, realDiff2) = compare_ssim(imageA[:, :, 2], realImageB[:, :, 2], full=True)
    # (score, diff) = compare_ssim(imageA, imageB, full=True)
    realDiff2 = (realDiff2 * 255).astype("uint8")
    #print("SSIM2: {}".format(realScore2))

    realDst1 = cv2.addWeighted(realDiff0, 0.333, realDiff1, 0.333, 0)
    realDst = cv2.addWeighted(realDst1, 1, realDiff2, 0.333, 0)

    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    # thresh = cv2.threshold(diff, 10, 255,
    #   cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    realThresh = cv2.threshold(realDst, 10, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # thresh0 = cv2.threshold(diff0, 10, 255,
    #    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # thresh1 = cv2.threshold(diff1, 10, 255,
    #    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # thresh2 = cv2.threshold(diff2, 10, 255,
    #    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # dst1 = cv2.addWeighted(thresh0,0.333,thresh1,0.333,0)
    # dst = cv2.addWeighted(dst1,0.333,thresh2,0.333,0)

    # contours
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    #	cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)

    # to close holes need masks with dilation
    realKernal = np.ones((3, 3), np.uint8)
    realDiletion = cv2.dilate(realThresh, realKernal, iterations=3)
    # erosia
    realErosion = cv2.erode(realDiletion, realKernal, iterations=3)

    cv2.namedWindow('realErosion', cv2.WINDOW_NORMAL)
    cv2.imshow('realErosion', realErosion)

    #cv2.namedWindow('diletion', cv2.WINDOW_NORMAL)
    #cv2.imshow('diletion', diletion)

    #cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
    #cv2.imshow('thresh', thresh)
    return realErosion

if __name__ == '__main__':
    main()
