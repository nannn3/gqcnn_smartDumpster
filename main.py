import pdb
from Astra import Astra
from Detection import detector
import cv2 as cv
import os
from autolab_core import ColorImage, DepthImage, RgbdImage, YamlConfig, CameraIntrinsics
from gqcnn.grasping import RgbdImageState, FullyConvolutionalGraspingPolicyParallelJaw
import time

# Initialize policy
config = YamlConfig(os.path.join(os.path.dirname(os.path.realpath(__file__)), "cfg/examples/fc_gqcnn_pj.yaml"))
policy_config = config["policy"]
policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_config)

def invokeDexNet(color, depth, segmask):
    """
    Invokes DexNet grasping policy on a given RGB-D image with segmentation mask.

    Parameters:
        color (numpy.ndarray): Color image.
        depth (numpy.ndarray): Depth image.
        segmask (autolab_core.BinaryImage): Segmentation mask.

    Returns:
        action: Grasping action.
    """
    color_im = ColorImage(color)
    depth_im = DepthImage(depth).inpaint()

    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    camera_intr = CameraIntrinsics.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Astra/Astra_IR.intr"))
    state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)
    policy_start = time.time()
    action = policy(state)
    print('\n Planning took %.3f sec' % (time.time() - policy_start))
    return action


def draw_grasp(action, im):
    '''Draws a rectangle and circles on the image
        Params:
            action: obj: action
            im: np.ndarray: Representing Image to draw on
        Returns: png to be shown
    '''
    foo = action.grasp.feature_vec
    p1 = (int(foo[0]), int(foo[1]))
    p2 = (int(foo[2]), int(foo[3]))
    depth = foo[4]
    '''
    Write to output file
    centroidX=action.grasp.center.vector[0]
    centroidY=action.grasp.center.vector[1]
    outfile = open("../../franky/franky/items/Items_Rot_Dep.txt","a")
    outdict={"X_0":centroidX,"Y_0":centroidY,"X_1":p1[0],"Y_1":p1[1],"X_2":p2[0],"Y_2":p2[1],"Dep":depth}
    outfile.write(str(outdict)+"\n")
    '''
    
    
    
    # Draw rectangle
    # This seemingly unessecary copy makes it work. If you don't copy, it won't. See SO link for more info
    im = im.copy() # https://stackoverflow.com/questions/23830618/python-opencv-typeerror-layout-of-the-output-array-incompatible-with-cvmat
    im_rec = cv.rectangle(im, p1, p2, (255, 0, 0), 1)
    
    # Draw circles at p1 and p2
    radius = 3  # Adjust the radius of the circles as needed
    thickness = 2  # Adjust the thickness of the circles as needed
    im_rec = cv.circle(im_rec, p1, radius, (0, 0, 255), thickness)
    im_rec = cv.circle(im_rec, p2, radius, (0, 0, 255), thickness)
    
    # Add depth text above the rectangle
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text = f"Depth: {depth:.2f}"
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_position = (p1[0], p1[1] - text_size[1] -25)  # Position text above the rectangle
    im_rec = cv.putText(im_rec, text, text_position, font, font_scale, (255, 255, 255), font_thickness)
    
    return im_rec

if __name__ == "__main__":
    # Setup camera
    camera = Astra.Astra()
    camera.start()

    # Define mouse click event
    posList = []
    runflag = False

    def onMouse(event, x, y, flags, param):
        """
        Mouse click event handler.

        Records mouse click positions for object detection.

        Parameters:
            event: Type of mouse event.
            x (int): x-coordinate of the mouse click.
            y (int): y-coordinate of the mouse click.
            flags: Flags indicating the state of the mouse buttons.
            param: Additional parameters.
        """
        global posList, runflag
        if event == cv.EVENT_LBUTTONDOWN:
            posList.append((x, y))
            runflag = True

    # Assign mouse click to the windows
    cv.namedWindow('color')
    cv.setMouseCallback('color', onMouse)
    cv.namedWindow('binary_image')
    cv.setMouseCallback('binary_image', onMouse)
    cv.namedWindow('depth')
    cv.setMouseCallback('depth', onMouse)

    # Set up object detector
    detector = detector.Detector("Detection/example_config.json")

    # Main event loop
    while 1:
        color, depth = camera.frames()
        cv.imshow("color", color)
        cv.imshow('depth', depth)
        contours, full_binary_image = detector.detect_objects(color, depth)
        cv.imshow("binary_image", full_binary_image._image_data())
        if runflag:
            runflag = False
            containing_contour = detector.find_contour_near_point(contours, posList[0])
            posList.pop(0)
            if containing_contour is None:
                print("No object found")
            else:
                single_obj_bin_im = full_binary_image.contour_mask(containing_contour)
                action = invokeDexNet(color, depth, single_obj_bin_im)
                if policy_config["vis"]["final_grasp"]:
                    im = draw_grasp(action,color)
                    cv.imshow("Planned grasp",im)

        cv.waitKey(1)

