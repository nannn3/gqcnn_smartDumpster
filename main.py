import pdb
import cv2 as cv
import os
from autolab_core import ColorImage, DepthImage, RgbdImage, YamlConfig, CameraIntrinsics, PointCloud
from gqcnn.grasping import RgbdImageState, FullyConvolutionalGraspingPolicyParallelJaw
import time
import numpy as np

from Astra import Astra
from Detection import detector
from pixelinput.pixel_input import PixelInput

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
    depth_im = DepthImage(depth).inpaint(1)
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    camera_intr = CameraIntrinsics.load(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Astra/Astra_IR.intr"))
    state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)
    
    policy_start = time.time()
    action = policy(state)
    
    print('\n Planning took %.3f sec' % (time.time() - policy_start))
    return action


def draw_grasp(action, im,depth_at_x_y=None):
    '''Draws a rectangle and circles on the image
        Params:
            action: obj: action
            im: np.ndarray: Representing Image to draw on
        Returns: png to be shown
    '''
    foo = action.grasp.feature_vec
    p1 = (int(foo[0]), int(foo[1]))
    p2 = (int(foo[2]), int(foo[3]))
    if depth_at_x_y:
        depth = depth_at_x_y
    else:
        depth = foo[4]
    
    # Write to output file
    centroidX=action.grasp.center.vector[0]
    centroidY=action.grasp.center.vector[1]
    outfile = open("../../franky/franky/items/Items_Rot_Dep.txt","a")
    outdict={"X_0":centroidX,"Y_0":centroidY,"X_1":p1[0],"Y_1":p1[1],"X_2":p2[0],"Y_2":p2[1],"Dep":depth}
    outfile.write(str(outdict)+"\n")
    
    
    # This copy makes it work.
    im = im.copy() # https://stackoverflow.com/questions/23830618/python-opencv-typeerror-layout-of-the-output-array-incompatible-with-cvmat
    
    
    # Draw rectangle
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
    camera_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Astra')
    color_intr_file = os.path.join(camera_file_path,'Astra_Color.intr')
    ir_intr_file = os.path.join(camera_file_path,'Astra_IR.intr')
    # Initialize the camera
    camera = Astra.Astra(color_intr_path = color_intr_file, ir_intr_path = ir_intr_file)
    camera.start()
    pixel_input = PixelInput()

    # Set up object detector
    detector = detector.Detector("Detection/example_config.json")
    theta = -.0471 * (np.pi/180)
    phi = -.0047 *(np.pi/180)


    R = np.array([[np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)*np.sin(phi),0],
                 [0,np.cos(theta),-np.sin(theta),0],
                 [-np.sin(phi),np.sin(theta)*np.cos(phi),np.cos(theta)*np.cos(phi),0],
                 [0,0,0,1]])
    # Main event loop
    while 1:
        color, depth = camera.frames()
        depth = camera.transform_image(depth,R)
        depth_color = camera.depth_to_color(depth)
        detector.find_calibration_cubes(color,depth)
        detector.draw_cube_points(color)
        cv.imshow("color", color)
        cv.imshow('depth', depth_color)
       
        contours, full_binary_image,*_ = detector.detect_objects(color, depth)
        cv.imshow("binary_image", full_binary_image._image_data())

        x,y = pixel_input.get_pixel_coordinates()
        if x is not None and y is not None:
            depth_at_x_y = depth[y][x]
            print(f'depth at ({x},{y}) is {depth_at_x_y}')
            containing_contour = detector.find_contour_near_point(contours,(x,y))
            if containing_contour is None:
                print("No object found")
            else:
                single_obj_bin_im = full_binary_image.contour_mask(containing_contour)
                action = invokeDexNet(color, depth, single_obj_bin_im)
                if policy_config["vis"]["final_grasp"]:
                    im = draw_grasp(action,color)#,depth_at_x_y)

                    cv.imshow("Planned grasp",im)
            

        cv.waitKey(1)

