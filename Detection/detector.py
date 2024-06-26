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
        self.load_cfg(config_file)

    def get_cfg(self, key):
        """
        Get a configuration value.

        Args:
            key (str): The key for the configuration value.

        Returns:
            The configuration value.

        Raises:
            KeyError: If the key is not found in the configuration.
        """
        try:
            return self.cfg[key]
        except KeyError:
            raise KeyError(f"Key '{key}' not found in configuration.")

    def detect_objects(self, color, depth):
        """
        Detect objects in the given images.

        Args:
            color (np.ndarray): The color image.
            depth (np.ndarray): The depth image.

        Returns:
           contours (ls): [cv2.contours]
        """
        color_im = ColorImage(color)
        depth_im = DepthImage(depth)        
        filtered_depth_im = self.create_threshold_im(depth_im)
        foreground_mask = self.create_foreground_mask(color_im)
        #overlayed_bin_ims = foreground_mask.mask_binary(filtered_depth_im)
        binary_im_filtered = self.filter_im(foreground_mask, self.get_cfg('morphological_filter_size'))        
        contours = binary_im_filtered.find_contours(min_area=self.get_cfg('min_contour_area'), max_area=self.get_cfg('max_contour_area'))
        return contours,binary_im_filtered
        
    def create_foreground_mask(self,color_im):
        """
        Create a foreground mask from a color image.

        Args:
            color_im (ColorImage): The color image.

        Returns:
            BinaryImage: The foreground mask.
        """
        mask_threshold = self.get_cfg('foreground_mask_threshold')
        foreground_mask = color_im.foreground_mask(mask_threshold,ignore_black = True )
        return foreground_mask
        
    def create_threshold_im(self,depth_im):
        """
        Create a thresholded binary image from a depth image.
        Things closer than 'depth_threshold_min' or further than 'depth_threshold_max' are masked out
        Args:
            depth_im (DepthImage): The depth image.

        Returns:
            right_depth_mask : The thresholded binary image.

        Raises:
            ValueError: If if depth_threshold_min is greater than or equal to depth_threshold_max.
        """
        depth_min = self.get_cfg('depth_threshold_min')
        depth_max = self.get_cfg('depth_threshold_max')
        if depth_max <= depth_min:
            raise ValueError("depth_threshold_min must be smaller than depth_threshold_max")
        binary_im = depth_im.threshold(depth_min,depth_max)
        depth_mask = binary_im.invalid_pixel_mask() # This creates a binary image where things outside the depth thresholds are white, so it needs to be inverted 
        #right_depth_mask = depth_mask.inverse()
        return depth_mask
        
    def filter_im(self, im, w):
        """
        Filter an image using morphological closing.

        Args:
            binary_im (BinaryImage): The binary image to filter.
            w (int): The size of the filter.

        Returns:
            im_filtered: The filtered  image.
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

    def find_contour_near_point(self,contours,pt):
        '''
            Picks out the contour containing the point pt
            Args:
                contours: list of cv2 contour objects
                pt : (int x , int y)
            Returns:
                contour: cv2 contour object containing pt or None if not found
        '''
        x_click,y_click = pt
        for obj in contours:
            box = obj.bounding_box
            y,x = box.center
            width = box.width
            height = box.height
            if (x_click >= x - width/2 and x_click <= x + width/2) and (y_click >= y - height/2 and y_click <= y+ height/2):
                return obj
        return None

if __name__ == "__main__":
    from Astra import Astra
    # Example configuration dictionary with original values
    '''example_cfg = {
        'depth_threshold_min': 0.5,  # Minimum depth threshold for object detection
        'depth_threshold_max': 1.5,  # Maximum depth threshold for object detection
        'foreground_mask_threshold': 225,  # Threshold for foreground mask generation
        'morphological_filter_size': 5,  # Size of the morphological filter
        'min_contour_area': 50.0,  # Minimum contour area for object detection objs
        'max_contour_area': np.inf # Maximum contour area for object detection objs 
    }'''
    posList = []
    runflag = False
    def onMouse(event, x, y, flags, param):
        'captures mouse x and y when mouse clicks'
        global posList, runflag
        if event == cv.EVENT_LBUTTONDOWN:
            posList.append((x, y))
            runflag = True
    cv.namedWindow('color')
    cv.setMouseCallback('color', onMouse)
    
    # Save example configuration to a JSON file
    '''with open("example_config.json", "w") as f:
        json.dump(example_cfg, f)   
    '''
    # Create detector instance with example configuration
    detector = Detector("example_config.json")

    # camera initialization
    camera = Astra.Astra()
    camera.start()
    
    while True:
        color, depth = camera.frames()
        depth_im = DepthImage(depth)
        threshold_mask = detector.create_threshold_im(depth_im)
        cv.imshow('threshold',threshold_mask._image_data())
        cv.imshow("color", color)
        if runflag:
            # Get segmask of object nearest mouseclick
            runflag = False
            contours,bin_im  = detector.detect_objects(color, depth)
            cv.imshow('all_obj',bin_im._image_data())
            cv.waitKey(1)
                  
            containing_contour = detector.find_contour_near_point(contours,posList[0])
            posList.pop(0)
            if containing_contour is None:
                print("no contour found")
            else:
                single_obj_bin_im = bin_im.contour_mask(containing_contour)
                cv.imshow('result',single_obj_bin_im._image_data())
            
             
        cv.waitKey(1)
        
       
