import numpy as np
import cv2 as cv
import os 

#define calibration pattern size:
pattern_size = (9,7)
square_size = 22.0

objp = np.zeros((np.prod(pattern_size),3),np.float32)
objp[:,:2] = np.indices(pattern_size).T.reshape(-1,2)
objp *= square_size

objpoints = []
imgpoints = []

# read images
image_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Calibration_Pics')

images = [cv.imread(os.path.join(image_folder,f'image{i}.png')) for i in range (1,2)]

for img in images:
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,corners = cv.findChessboardCorners(gray,pattern_size,None)
    cv.imshow('img',img)
    cv.waitKey(1)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001))
        imgpoints.append(corners2)
ret, mtx, dist, rvecs,tvecs = cv.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)

R,_ = cv.Rodrigues(rvecs[0])

print("rotation Vector:\n",R)
print("translation Vector:\n",tvecs)
print("matrix:\n",mtx)
