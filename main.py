from Astra import Astra
from Detection import detector
import cv2 as cv
import os
from autolab_core import BinaryImage, ColorImage, DepthImage, RgbdImage

import subprocess
from gqcnn.grasping import RgbdImageState,FullyConvolutionalGraspingPolicyParallelJaw
ImageFolder = 'ImTestPics'

#initialize policy:
policy = FullyConvolutionalGraspingPolicyParallelJaw(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cfg/examples/fc_gqcnn_pj.yaml"))

def invokeDexNet(color,depth,segmask):
    '''
    for now we can probably just directly invoke it from this python script by saving color.png
    segmask.png, and depth.npy and calling examples/policy.py.
    eventually, this should be rebuilt as our own implementation of policy.py
    '''
    #segmask_file = os.path.join(ImageFolder,'segmask.png')
    #color_file = (os.path.join(ImageFolder,'color.png'))   
    #depth_file = (os.path.join(ImageFolder,'depth.npy'))
    
    color_im = ColorImage(color)
    depth_im = DepthImage(depth).inpaint()

    rgbd_im = RgbdImage.from_color_and_depth(color_im,depth_im)
    camera_intr = CameraInstrinsics.load(os.path.join(os.path.dirname(os.path.realpath(__file__)),"Astra/Astra_IR.intr"))
    state = RgbdImageState(rgbd_im, camera_intr, segmask = segmask)
    
    action = policy(state)
    print(action.grasp.feature_vec)
    #segmask.save(segmask_file)
    #color_im.save(color_file)
    #depth_im.save(depth_file)
    
    command = ["python","examples/policy.py","FC-GQCNN-4.0-PJ","--fully_conv","--color_image",color_file,"--depth_image",depth_file,"--segmask",segmask_file]
    subprocess.run(command)

if __name__=="__main__":
    #Setup camera:
    camera = Astra.Astra()
    camera.start()
    
    policy = FullyConvolutionalGraspingPolicyParallelJaw(os.path.join(os.path.dirname(os.path.realpath(__file__)),"cfg/examples/fc_gqcnn_pj.yaml"))


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
    cv.namedWindow('depth')
    cv.setMouseCallback('depth',onMouse)
    #set up detector
    detector = detector.Detector("Detection/example_config.json")
    #main event loop
    while 1:
        color,depth = camera.frames()
        cv.imshow("color",color)
        cv.imshow('depth',depth)
        contours,full_binary_image = detector.detect_objects(color,depth)
        cv.imshow("binary_image",full_binary_image._image_data())
        if runflag:
            #itemfile=open("../../franky/franky/items/Pending_Items_Camera.txt", "a")
            #itemfile.write(str(posList[0][0]) + "," + str(posList[0][1])+"|\n")
            #itemfile.close()
            #STARTfile=open("../../franky/franky/items/START.txt", "a")
            #STARTfile.write("START")
            #STARTfile.close()
            runflag = False
            containing_contour = detector.find_contour_near_point(contours, posList[0])
            posList.pop(0)
            if containing_contour is None:
                print("No object found")
            else:
                single_obj_bin_im = full_binary_image.contour_mask(containing_contour)
                invokeDexNet(color,depth,single_obj_bin_im)
        cv.waitKey(1)
