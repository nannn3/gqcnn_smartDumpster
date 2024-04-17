import scipy.ndimage.morphology as snm
import numpy as np
from autolab_core import ColorImage, DepthImage, BinaryImage
import cv2 as cv
import json

class Detector:
    def __init__(self, config_file):
        """
        Initialize the Detector with a configuration file.

        Args:
            config_file (str): Path to the configuration file.
        """
        try:
            with open(config_file, 'r') as f:
                self.cfg = json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            self.cfg = {}

    def get_cfg(self, key):
        """
        Get a configuration value.

        Args:
            key (str): The key for the configuration value.

        Returns:
            The configuration value, or None if the key is not found.
        """
        return self.cfg.get(key, None)

    def set_cfg(self, key, value):
        """
        Set a configuration value.

        Args:
            key (str): The key for the configuration value.
            value: The value to set.
        """
        self.cfg[key] = value

    def remove_cfg(self, key):
        """
        Remove a key from the configuration.

        Args:
            key (str): The key to remove.
        """
        if key in self.cfg:
            del self.cfg[key]

    def detect_objects(self, color, depth):
        """
        Detect objects in the given images.

        Args:
            color (np.ndarray): The color image.
            depth (np.ndarray): The depth image.

        Returns:
            contours (ls) : [cv2.contours] of all detected objects in frame.
        """
        color_im = ColorImage(color)
        depth_im = DepthImage(depth)

        filtered_depth_im = depth_im.threshold(.5,1)
        bin_im = color_im.foreground_mask(50, ignore_black=False)
        depth_mask = filtered_depth_im.invalid_pixel_mask()
        right_depth_mask = depth_mask.inverse()
        final_obj_mask = bin_im.mask_binary(right_depth_mask)
        
        binary_im_filtered = self.filter_im(final_obj_mask, 5)  #TODO Replace 5 with non-arbitrary value
        contours = binary_im_filtered.find_contours(min_area=50.0, max_area=np.inf)
        if len(contours) == 0:
            raise ValueError ("No objects found in frame")        
        
        return contours

    def filter_im(self, im, w):
        """
        Filter a binary image using morphological closing.

        Args:
            binary_im (BinaryImage): The binary image to filter.
            w (int): The size of the filter.

        Returns:
            BinaryImage: The filtered binary image.
        """
        y, x = np.ogrid[-w / 2 + 1 : w / 2 + 1, -w / 2 + 1 : w / 2 + 1]
        mask = x * x + y * y <= w / 2 * w / 2
        filter_struct = np.zeros([w, w]).astype(np.uint8)
        filter_struct[mask] = 1
        im_filtered = im.apply(
            snm.grey_closing, structure=filter_struct
        )
        return im_filtered

    def save_object_data(self, rgbd_detection_objects):
        """
        Save object data to an image file. This is a placeholder method.

        Args:
            rgbd_detection_objects: A list of RGBD.detection objects.
        """
        # TODO: Implement the method to save object data to an image file
        pass

    def save_cfg(self, filename):
        """
        Save the current configuration to a JSON file.

        Args:
            filename (str): The file name to save the configuration to.
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.cfg, f)
        except Exception as e:
            print(f"Error saving config file: {e}")

    def load_cfg(self, filename):
        """
        Load a configuration from a JSON file.

        Args:
            filename (str): The file name to load the configuration from.
        """
        try:
            with open(filename, 'r') as f:
                self.cfg = json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")

    def reset_cfg(self):
        """
        Reset the configuration to an empty dictionary.
        """
        self.cfg = {}

if __name__ == "__main__":
    detector = Detector("config.json")
    camera = Astra.Astra()
    camera.start()
    while True:
        color, depth = camera.frames()
        contours = detector.detect_objects(color, depth)
        #TODO Hardcode point for test here? To be replaced by mouseclick event in later version
        cv.imshow("color", color)
        cv.waitKey(1)
