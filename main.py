from Astra import Astra
from Detection import detector
import cv2 as cv
import os
from autolab_core import BinaryImage, ColorImage, DepthImage

ImageFolder = 'ImTestPics'

def invokeDexNet(color,depth,segmask):
    '''
    for now we can probably just directly invoke it from this python script by saving color.png
    segmask.png, and depth.npy and calling examples/policy.py.
    eventually, this should be rebuilt as our own implementation of policy.py
    '''
    segmask.save(os.path.join(ImageFolder,'segmask.png'))
    color_im = ColorImage(color)
    color_im.save(os.path.join(ImageFolder,'color.png'))
    depth_im = DepthImage(depth)
    depth_im.save(os.path.join(ImageFolder,'depth.npy'))

    

if __name__=="__main__":
    #Setup camera:
    camera = Astra.Astra()
    camera.start()
    # Check if the folder exists
    if not os.path.exists(ImageFolder):
        # If the folder doesn't exist, create it
        os.makedirs(ImageFolder)
    
    #define mouse click event:
    posList = []
    runflag = False
    def onMouse(event,x,y,flags,param):
        global posList, runflag
        if event == cv.EVENT_LBUTTONDOWN:
            posList.append((x,y))
            runflag = True
    #Assign mouseclick to the color and binary windows
    cv.namedWindow('color')
    cv.setMouseCallback('color',onMouse)
    cv.namedWindow('binary_image')
    cv.setMouseCallback('binary_image',onMouse)
    #set up detector
    detector = detector.Detector("Detection/example_config.json")
    #main event loop
    while 1:
        color,depth = camera.frames()
        cv.imshow("color",color)
        contours,full_binary_image = detector.detect_objects(color,depth)
        cv.imshow("binary_image",full_binary_image._image_data())
        if runflag:
            runflag = False
            containing_contour = detector.find_contour_near_point(contours, posList[0])
            posList.pop(0)
            if containing_contour is None:
                print("No object found")
            else:
                single_obj_bin_im = full_binary_image.contour_mask(containing_contour)
                cv.imshow('result',single_obj_bin_im._image_data())
                invokeDexNet(color,depth,single_obj_bin_im)
        cv.waitKey(1)
