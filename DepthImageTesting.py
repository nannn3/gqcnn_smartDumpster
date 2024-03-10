import sys
CV_PATH = '/usr/local/lib/python3.7/site-packages'
if CV_PATH not in sys.path:
    sys.path.append(CV_PATH)
import cv2 as cv

cap = cv.VideoCapture(cv.CAP_OPENNI2_ASTRA)
print(cv.CAP_OPENNI2_ASTRA)
if not cap.isOpened():
    exit("depth Camera not opened")
print('depth opened')

cap2 = cv.VideoCapture('/dev/bus/usb/001/010',cv.CAP_ORBSENSOR)
if not cap2.isOpened():
    exit("rgb camera not opened")

if cap.grab():
    ret,image = cap.retrieve(None,cv.CAP_OPENNI_DEPTH_MAP)
    if not ret:
        print("OPENNI_DEPTH_MAP bad")
    else:
        cv.imwrite('CAP_OPENNI_DEPTH_MAP.png',image)
if(cap.grab()):
    ret,image = cap.retrieve(None,cv.CAP_OBSENSOR_BGR_IMAGE)
    if not ret:
        print("OBSENSOR_BGR_IMAGE bad")
    else:
        cv.imwrite('OBSENSOR_BGR_IMAGE.png',image)
if(cap.grab()):
    ret,image = cap.retrieve(None,cv.CAP_OBSENSOR_DEPTH_MAP)
    if not ret:
        print("OBSENSOR_DEPTH_MAP bad")
    else:
        cv.imwrite('OBSENSOR_DEPTH_MAP.png',image)

cap.release()
