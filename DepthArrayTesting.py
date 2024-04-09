import cv2 as cv
import pdb
import time
import numpy as np
import json
import os
from autolab_core import (YamlConfig, Logger, BinaryImage, CameraIntrinsics,
                          ColorImage, DepthImage, RgbdImage,constants, detector)
from visualization import Visualizer2D as vis

from gqcnn.grasping import (RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)
from gqcnn.utils import GripperMode
from Astra import Astra

if __name__=="__main__":
    det_cfg = {"foreground_mask_tolerance":1,
            "min_contour_area":50,
            "max_contour_area":100000,
            "filter_dim":1
            }
    dev = Astra()
    dev.start()
    camera_intr_filename = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "data/calib/primesense/primesense.intr")
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"models/FC-GQCNN-4.0-PJ")
    model_config = json.load(open(os.path.join(model_path, "config.json"),
                                  "r"))

    try:
        gqcnn_config = model_config["gqcnn"]
        gripper_mode = gqcnn_config["gripper_mode"]
    except KeyError:
        gqcnn_config = model_config["gqcnn_config"]
        input_data_mode = gqcnn_config["input_data_mode"]
        if input_data_mode == "tf_image":
            gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
        elif input_data_mode == "tf_image_suction":
            gripper_mode = GripperMode.LEGACY_SUCTION
        elif input_data_mode == "suction":
            gripper_mode = GripperMode.SUCTION
        elif input_data_mode == "multi_suction":
            gripper_mode = GripperMode.MULTI_SUCTION
        elif input_data_mode == "parallel_jaw":
            gripper_mode = GripperMode.PARALLEL_JAW
        else:
            raise ValueError(
                "Input data mode {} not supported!".format(input_data_mode))

    config_filename = os.path.join(
                    os.path.dirname(os.path.realpath(__file__)),
                    "cfg/examples/fc_gqcnn_pj.yaml")

    config = YamlConfig(config_filename)
    segmask= None
    inpaint_rescale_factor = config["inpaint_rescale_factor"]
    policy_config = config["policy"]
    # Setup sensor.
    camera_intr = CameraIntrinsics.load(camera_intr_filename)

    # Read images.
    # pdb.set_trace()
    color_frame,depth_frame = dev.frames()
    color_im = ColorImage(color_frame)
    depth_frame = depth_frame * constants.MM_TO_METERS
    depth_im = DepthImage(depth_frame)
    det = detector.RgbdForegroundMaskDetector()
    objs = det.detect(color_im,depth_im,det_cfg)
    for indx, obj in enumerate(objs):
        obj.binary_im.save(f'{indx}_seg_img.png')
        obj.color_thumbnail.save(f'{indx}_color_thumb.png')
    # Inpaint.
    depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
    if "input_images" in policy_config["vis"] and policy_config["vis"][
        "input_images"]:
        vis.figure(size=(10, 10))
        num_plot = 1
        if segmask is not None:
            num_plot = 2
            vis.subplot(1, num_plot, 1)
            vis.imshow(depth_im)
        if segmask is not None:
            vis.subplot(1, num_plot, 2)
            vis.imshow(segmask)
        vis.show()

    # Create state.
    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)
    policy_type = "cem"
    '''
    if "type" in policy_config:
        policy_type = policy_config["type"]
        if policy_type == "ranking":
            policy = RobustGraspingPolicy(policy_config)
        elif policy_type == "cem":
            policy = CrossEntropyRobustGraspingPolicy(policy_config)
        else:
            raise ValueError("Invalid policy type: {}".format(policy_type))
     '''
    policy = CrossEntropyRobustGraspingPolicy(policy_config)
   # Query policy.
   
    policy_start = time.time()
    action = policy(state)
    logger.info("Planning took %.3f sec" % (time.time() - policy_start))

    # Vis final grasp.
    if policy_config["vis"]["final_grasp"]:
        vis.figure(size=(10, 10))
        vis.imshow(rgbd_im.depth,
                   vmin=policy_config["vis"]["vmin"],
                   vmax=policy_config["vis"]["vmax"])
        vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
            action.grasp.depth, action.q_value))
        vis.show()
