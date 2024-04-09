import cv2 as cv
import time
import json
import os
from autolab_core import (
    YamlConfig, Logger, CameraIntrinsics, ColorImage, DepthImage,
    RgbdImage, constants, detector
)
from visualization import Visualizer2D as vis
from gqcnn.grasping import (
    CrossEntropyRobustGraspingPolicy, RgbdImageState
)
from gqcnn.utils import GripperMode
from Astra import Astra

def load_configuration():
    """Load configuration parameters for object detection.

    Returns:
        dict: Dictionary containing detection configuration parameters.
    """
    det_cfg = {
        "foreground_mask_tolerance": 1,
        "min_contour_area": 50,
        "max_contour_area": 100000,
        "filter_dim": 1
    }
    return det_cfg

def load_model_configuration():
    """Load gripper mode configuration from model.

    Returns:
        int: Gripper mode constant.
    """
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models/FC-GQCNN-4.0-PJ")
    model_config = json.load(open(os.path.join(model_path, "config.json"), "r"))
    try:
        gqcnn_config = model_config["gqcnn"]
        gripper_mode = gqcnn_config["gripper_mode"]
    except KeyError:
        gqcnn_config = model_config["gqcnn_config"]
        input_data_mode = gqcnn_config["input_data_mode"]
        gripper_mode = get_gripper_mode(input_data_mode)
    return gripper_mode

def get_gripper_mode(input_data_mode):
    """Map input data mode to gripper mode.

    Args:
        input_data_mode (str): Input data mode.

    Returns:
        int: Gripper mode constant.
    """
    modes = {
        "tf_image": GripperMode.LEGACY_PARALLEL_JAW,
        "tf_image_suction": GripperMode.LEGACY_SUCTION,
        "suction": GripperMode.SUCTION,
        "multi_suction": GripperMode.MULTI_SUCTION,
        "parallel_jaw": GripperMode.PARALLEL_JAW
    }
    if input_data_mode not in modes:
        raise ValueError(f"Input data mode {input_data_mode} not supported!")
    return modes[input_data_mode]

def load_configuration_file():
    """Load YAML configuration file.

    Returns:
        YamlConfig: YAML configuration object.
    """
    config_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cfg/examples/fc_gqcnn_pj.yaml")
    return YamlConfig(config_filename)

def capture_frames(device):
    """Capture color and depth frames from the device.

    Args:
        device: Device object.

    Returns:
        tuple: A tuple containing ColorImage and DepthImage objects.
    """
    color_frame, depth_frame = device.frames()
    color_im = ColorImage(color_frame)
    depth_frame = depth_frame * constants.MM_TO_METERS
    depth_im = DepthImage(depth_frame)
    return color_im, depth_im

def detect_objects(detector, color_im, depth_im, det_cfg):
    """Detect objects in the scene.

    Args:
        detector: Object detector.
        color_im: ColorImage object.
        depth_im: DepthImage object.
        det_cfg (dict): Detection configuration parameters.

    Returns:
        list: List of detected objects.
    """
    return detector.detect(color_im, depth_im, det_cfg)

def save_images(objects):
    """Save segmented images and color thumbnails.

    Args:
        objects (list): List of detected objects.
    """
    for indx, obj in enumerate(objects):
        obj.binary_im.save(f'{indx}_seg_img.png')
        obj.color_thumbnail.save(f'{indx}_color_thumb.png')

def inpaint_depth_image(depth_im, inpaint_rescale_factor):
    """Inpaint depth image.

    Args:
        depth_im: DepthImage object.
        inpaint_rescale_factor (float): Inpaint rescale factor.

    Returns:
        DepthImage: Inpainted depth image.
    """
    return depth_im.inpaint(rescale_factor=inpaint_rescale_factor)

def visualize_images(depth_im, segmask, policy_config):
    """Visualize input images.

    Args:
        depth_im: DepthImage object.
        segmask: Segmentation mask (optional).
        policy_config (dict): Policy configuration parameters.
    """
    if "input_images" in policy_config["vis"] and policy_config["vis"]["input_images"]:
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

def setup_policy(config, policy_config):
    """Set up grasping policy.

    Args:
        config: Configuration object.
        policy_config (dict): Policy configuration parameters.

    Returns:
        CrossEntropyRobustGraspingPolicy: Grasping policy object.
    """
    policy = CrossEntropyRobustGraspingPolicy(policy_config)
    return policy

def query_policy(policy, state):
    """Query grasping policy.

    Args:
        policy: Grasping policy object.
        state: State object.

    Returns:
        Action: Action object.
    """
    policy_start = time.time()
    action = policy(state)
    logger.info("Planning took %.3f sec" % (time.time() - policy_start))
    return action

def visualize_grasp(rgbd_im, action, policy_config):
    """Visualize the planned grasp.

    Args:
        rgbd_im: RgbdImage object.
        action: Action object.
        policy_config (dict): Policy configuration parameters.
    """
    if policy_config["vis"]["final_grasp"]:
        vis.figure(size=(10, 10))
        vis.imshow(rgbd_im.depth, vmin=policy_config["vis"]["vmin"], vmax=policy_config["vis"]["vmax"])
        vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
        vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(action.grasp.depth, action.q_value))
        vis.show()

if __name__ == "__main__":

    # Set up camera and turn it on:
    device = Astra()
    device.start()
    
    # Load configs
    det_cfg = load_configuration()
    gripper_mode = load_model_configuration()
    config = load_configuration_file()
    camera_intr_filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/calib/primesense/primesense.intr")
    camera_intr = CameraIntrinsics.load(camera_intr_filename)

    # Get color and depth images 
    color_im, depth_im = capture_frames(device)
    
    # Generate list of all objects detected in the frame
    det = detector.RgbdForegroundMaskDetector()
    objects = detect_objects(det, color_im, depth_im, det_cfg)
    # Save the images
    save_images(objects)
    # Inpaint
    depth_im = inpaint_depth_image(depth_im, config["inpaint_rescale_factor"])

    segmask = None  # You may need to set segmask here

    visualize_images(depth_im, segmask, config["policy"])

    rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
    state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)

    policy = setup_policy(config, config["policy"])

    action = query_policy(policy, state)

    visualize_grasp(rgbd_im, action, config["policy"])
