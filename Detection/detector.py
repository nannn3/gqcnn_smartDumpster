import pdb
import scipy.ndimage as snm
import numpy as np
from autolab_core import ColorImage, DepthImage, BinaryImage, Contour
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
        #It might be a problem that tall orange and tall white are basically the same color, however the Y value of tall white will always be greater than that of tall orange
        self.cubes = {
                'Tall_Green':{'color':(88,90,89)},
                'Short_Yellow':{'color':(180,245,250)},
                'Tall_Orange':{'color':(252,250,248)},
                'Short_White':{'color':(252,255,248)},
                'Short_Green':{'color':(144,205,158)},
                'Tall_Yellow':{'color':(205,247,249)},
                'Short_Orange':{'color':(146,196,246)},
                'Tall_White':{'color':(253,253,253)}
                }
    def is_same_color(self,recorded_color,known_color):
        '''
        compares two tuples and checks if their colors are within a tolerance level
        '''
        if len(recorded_color) != len(known_color):
            raise ValueError("recorded color and known color should be the same size")

        tolerance = self.get_cfg('color_tolerance')
        for c1,c2 in zip(recorded_color,known_color):
            if abs(c1-c2) < tolerance:
                return False

        return True
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
        overlayed_bin_ims = foreground_mask.pixelwise_or(filtered_depth_im)
        binary_im_filtered = self.filter_im(overlayed_bin_ims, self.get_cfg('morphological_filter_size'))        
        
        contours = binary_im_filtered.find_contours(min_area=self.get_cfg('min_contour_area'), max_area=self.get_cfg('max_contour_area'))
        #tops_of_objects = self.find_object_tops(depth, contours)
        #tops_of_objects = BinaryImage(tops_of_objects)
        #contours = tops_of_objects.find_contours(min_area = self.get_cfg('min_contour_area'), max_area = self.get_cfg('max_contour_area'))
        #return contours,tops_of_objects
        return contours,binary_im_filtered,foreground_mask,filtered_depth_im
   
    def find_object_tops(self, depth_image, contours, threshold=.01):
        """
        Generates a binary image highlighting the tops of objects in a depth image based on given contours and a depth threshold.
        
        Parameters:
        - depth_image (numpy.ndarray): A 480x640 numpy array containing depth values where each element represents the depth away from the camera at that pixel.
        - contours (list of numpy.ndarray): A list of contours where each contour is represented as an array of points.
        - depth_threshold (float): A threshold specifying the maximum distance above the minimum depth within each contour to still be considered as the top of the object.
        
        Returns:
        - numpy.ndarray: A binary image of the same shape as `depth_image`, where the tops of objects are marked with 255 and other areas are 0.
        
        The function works by iterating over each provided contour, creating a mask for that contour, and identifying the minimum depth within the masked region. It then marks all areas within the defined depth threshold from the minimum depth as the tops of objects. The resulting binary image is a combination of these markings for all contours.
        """
        
        # Prepare the final binary image
        final_bin_im = np.zeros_like(depth_image, dtype=np.uint8)
        # Process each contour
        for contour in contours:
            contour = contour.boundary_pixels.reshape(-1,1,2).astype(np.int32)
            contour = contour[:, 0, ::-1] #Swap x, y coordinates
            mask = np.zeros_like(depth_image, dtype=np.uint8)
            cv.drawContours(mask, [contour], -1, 255, thickness=cv.FILLED)
            # Applying the mask to the depth image to get the ROI
            roi = np.where(mask == 255, depth_image, np.max(depth_image) + 1)
            roi[roi == 0] = np.inf #swap 0's with infinity to get a better min depth
            min_depth = np.min(roi)
            # Creating a mask for areas close to the minimum depth
            close_to_min = np.where((roi >= min_depth - threshold) & (roi <= min_depth + threshold), 255, 0)
            foo = BinaryImage(close_to_min.astype(np.uint8))
            # Combine the current mask with the final binary image
            final_bin_im = cv.bitwise_or(final_bin_im, close_to_min.astype(np.uint8))
        
        return final_bin_im

    def create_foreground_mask(self,color_im):
        """
        Create a foreground mask from a color image.

        Args:
            color_im (ColorImage): The color image.

        Returns:
            BinaryImage: The foreground mask.
        """
        mask_threshold = self.get_cfg('foreground_mask_threshold')
        foreground_mask = color_im.foreground_mask(mask_threshold, bgmodel=[5,5,5])
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
        right_depth_mask = depth_mask.inverse()
        return right_depth_mask
        
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

    def find_contour_near_point(self, contours, pt):
        """
        Finds the nearest contour to the given point.

        Args:
            contours (list of np.ndarray): List of cv2 contour objects.
            pt (tuple of int): Point (x, y) for which the nearest contour is to be found.

        Returns:
            np.ndarray: The nearest contour to the point or None if there are no contours.
        """
        min_distance = float('inf')
        nearest_contour = None
        x_click,y_click = pt
        for contour in contours:
            '''
            contour = contour.boundary_pixels.reshape(-1,1,2).astype(np.int32)
            # Compute distance from the point to the contour
            distance = cv.pointPolygonTest(contour, pt, True)

            # Check if this contour is closer than what we have seen so far
            if distance < min_distance:
                min_distance = distance
                nearest_contour = contour
            '''
            box = contour.bounding_box
            y,x = box.center
            dist = np.sqrt((x_click-x)**2 + (y_click - y)**2)
            if dist < min_distance:
                min_distance = dist
                nearest_contour = contour
        
        return nearest_contour

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
    camera = Astra()
    camera.start()
    theta = -0.0471
    theta *= (np.pi/180)
    phi = -.0047 *(np.pi/180)
    '''
    T = np.array([
        [1,0,0,0],
        [0,np.cos(theta),-np.sin(theta),0],
        [0,np.sin(theta),np.cos(theta),0],
        [0,0,0,1]
        ]
        )
    '''
    T = np.array([
        [np.cos(phi),np.sin(phi)*np.sin(theta),np.sin(phi)*np.cos(theta),0],
        [0,np.cos(theta),-np.sin(theta),0],
        [-np.sin(phi),np.sin(theta)*np.cos(phi),np.cos(theta)*np.cos(phi),0],
        [0,0,0,1]
        ])
    cal = True
    while True:
        color, depth = camera.frames()
        depth = camera.transform_image(depth,T)
        depth_im = DepthImage(depth)

        contours,tops,color_mask,depth_mask = detector.detect_objects(color, depth)
        cv.imshow('color_mask',color_mask._image_data())
        cv.imshow('depth_mask',depth_mask._image_data())
        cv.imshow('tops',tops._image_data())
        cv.imshow("color", color)
        # Find mean color of calibration cubes:
        pts =[(276,77),(332,92),(403,72),(455,95),(263,340),(351,330),(427,340),(506,347)]
        for pt in pts:
            if not cal:
                break
            contour = detector.find_contour_near_point(contours,pt)
            contour = contour.boundary_pixels.reshape(-1,1,2).astype(np.int32)
            contour = contour[:, 0, ::-1] #Swap x, y coordinates
            mask = np.zeros_like(depth,dtype=np.uint8)
            mask = cv.drawContours(mask,[contour],-1,255,cv.FILLED)
            cv.imshow('mask',mask)
            mean_color = cv.mean(color,mask=mask)

            print("Color at point",pt,"is :",mean_color)
        cal = False
        if runflag:
            # Get segmask of object nearest mouseclick
            runflag = False
            cv.waitKey(1)
                  
            containing_contour = detector.find_contour_near_point(contours,posList[0])
            posList.pop(0)
            if containing_contour is None:
                print("no contour found")
            else:
                single_obj_bin_im = tops.contour_mask(containing_contour)
                cv.imshow('result',single_obj_bin_im._image_data())
            
             
        cv.waitKey(1)
        
       
