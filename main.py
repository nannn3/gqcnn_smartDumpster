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

    # Assign mouse click to the color and binary windows
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
                invokeDexNet(color, depth, single_obj_bin_im)
        cv.waitKey(1)

