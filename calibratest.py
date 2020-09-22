
# Useful links
# http://www.pygame.org/wiki/OBJFileLoader
# https://rdmilligan.wordpress.com/2015/10/15/augmented-reality-using-opencv-opengl-and-blender/
# https://clara.io/library

# TODO -> Implement command line arguments (scale, model and object to be projected)
#      -> Refactor and organize code (proper funcition definition and separation, classes, error handling...)


# import the necessary packages
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
    #projectedPoints = cv2.projectPoints(np.float32(objectPoints), rvec, tvec, np.float32(a), np.float32(distCoeffs), np.float32(imagePoints))
    #points = projectedPoints[0]


    #print(status)
    #print(rvec)
    #print(tvec)
    #print(imagePoints)
    #print(points)

    #myObjPoint = np.array([102, 0, 35])
    #myProjectedPoint = cv2.projectPoints(np.float32(myObjPoint), rvec, tvec, np.float32(a), np.float32(distCoeffs))
    #myPrPoint = 0.5 * myProjectedPoint[0]
    #print(myPrPoint)

    dir_name = os.getcwd()
    #frame = cv2.imread(os.path.join(dir_name, 'snapshots/123.jpg'))
    frame = cv2.imread(os.path.join(dir_name, 'backsnap/nonempty.jpg'))
    # Load 3D model from OBJ file
    #obj = OBJ(os.path.join(dir_name, 'models/fox.obj'), swapyz=True)
    obj = OBJ(os.path.join(dir_name, 'models/calibration.obj'), swapyz=False)
    #obj = OBJ(os.path.join(dir_name, 'models/Baby_Yoda.obj'), swapyz=True)
    verticesx2 = cv2.projectPoints(np.float32(obj.vertices), rvec, tvec, np.float32(a), np.float32(distCoeffs))
    vertices = 0.5 * verticesx2[0]

    # Using enumerate()
    # draw points
    #for i, val in enumerate(vertices):
    #  print (i, ",",val)
    #cv2.circle(frame,(int(val[0][0]), int(val[0][1])), 10, (0,0,255), -1)

    for i in enumerate(vertices):
        cv2.circle(frame, (int(vertices[0][0][i]), int(vertices[0][0][i])), 10, (0,0,255), -1)

    #cv2.circle(frame,(720, 913), 40, (0,0,255), -1)
    #cv2.circle(frame,(720, 913), 40, (0,0,255), -1)
    #cv2.circle(frame,(int(myPrPoint[0][0][0]), int(myPrPoint[0][0][1])), 40, (0,0,255), -1)
    #frame = render(obj, vertices, frame, color=False)

    #frame1 = cv2.imread(os.path.join(dir_name, 'backsnap/nonempty.jpg'))
    #cv2.namedWindow('reference', cv2.WINDOW_NORMAL)
    #cv2.imshow('reference', frame1)

    cv2.namedWindow('ProjectedModel', cv2.WINDOW_NORMAL)
    cv2.imshow('ProjectedModel', frame)


    cv2.waitKey()
    cv2.destroyAllWindows()
    return 0






if __name__ == '__main__':
    main()
