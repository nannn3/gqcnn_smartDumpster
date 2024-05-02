import numpy as np
import cv2 as cv
import os

# Define calibration pattern size:
pattern_size = (9, 7)
square_size = 22.0

# Prepare object points based on the known size of the chessboard:
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all images:
objpoints = []
imgpoints = []

# Read images
image_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Calibration_Pics')
images = [cv.imread(os.path.join(image_folder, f'image{i}.png')) for i in range(1, 2)]

for img in images:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        # If found, add object points, image points (after refining them)
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        
        # Draw and display the corners
        cv.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv.imshow('Detected Corners', img)
        
        # Wait until 'c' is pressed
        while True:
            if cv.waitKey(1) & 0xFF == ord('c'):
                break

cv.destroyAllWindows()  # Close the window after displaying all images

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

R, _ = cv.Rodrigues(rvecs[0])

print("Rotation Matrix:\n", R)
print("Translation Vector:\n", tvecs[0])
print("Camera Matrix:\n", mtx)
