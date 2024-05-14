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
        self.cubes = {
                'Tall_Green':{'color':(110,175,110)},
                'Short_Yellow':{'color':(200,250,250)},
                'Tall_Orange':{'color':(132,195,245)},
                'Short_White':{'color':(252,250,248)},
                'Short_Green':{'color':(155,230,155)},
                'Tall_Yellow':{'color':(235,247,249)},
                'Short_Orange':{'color':(157,225,246)},
                'Tall_White':{'color':(253,253,253)}
                }
    def draw_cube_points(self):
        for name,prop in self.cubes.items():
            try:
                contour = prop['contour']
                box = contour.bounding_box
                y,x = box.center
                print(f"{name} center point at ({x},{y})")
            except KeyError:
                print(f"{name} has no contour")

    def compare_color_to_cubes(self, color):
        """
        Compares a given color to predefined cube colors and returns the matching cube name.
        
        Args:
            color (tuple): The color to compare.
        
        Returns:
            str: Name of the cube if a match is found, otherwise None.
        """
        for cube_name, cube_props in self.cubes.items():
            if self.is_same_color(color, cube_props['color']):
                return cube_name
        return None
    
    def find_calibration_cubes(self,color_image,depth):
        '''
        Finds all the calibration cubes and adds their contour to the cube_prop dict
        Args:
            color_image (np.ndarray) the color image
            contours a list of Contour objects
        Raises:
            ValueError if there are less contours than cubes in the dict.
        '''
        contours = self.detect_objects(color_image,depth)[0]
        if len(contours) < len(self.cubes):
            raise ValueError("There should be at least as many contours as cubes")
        for contour in contours:
            color = self.get_color_from_contour(color_image,contour)
            cube_name = self.compare_color_to_cubes(color)
            if cube_name is None:
                continue
            cube_prop = self.cubes[cube_name]
            height = self.find_object_tops(depth,[contour])[1]
            if 'White' in cube_name:
                # white cubes have basically the same color, so we can use height to tell them apart
                # The smaller cube has a bigger height value (it's further from the camera)
                cube_name = 'Short_White'
                cube_prop = self.cubes[cube_name]
                short_white_height = cube_prop.get('height',None)
                if short_white_height is not None and height < short_white_height:
                    cube_name = 'Tall_White'
                    cube_prop = self.cubes[cube_name]
                elif short_white_height is not None and height > short_white_height:
                    pass #TODO Fix this later
                
                
            cube_prop['contour'] = contour
            cube_prop['height'] = height
            self.cubes[cube_name] = cube_prop
            print(f'contour added to {cube_name}')



                

    def get_color_from_contour(self, color_image, contour):
        """
        Extracts the mean color from the specified contour area in the given color image.
        
        Args:
            color_image (np.ndarray): The image from which color is extracted.
            contour (Contour): The contour within which color is measured.
        
        Returns:
            tuple: The mean color in (R, G, B) format.
        """
        mask = np.zeros_like(color_image[:, :, 2], dtype=np.uint8)
        contour = contour.boundary_pixels.reshape(-1, 1, 2).astype(np.int32)
        mask = cv.drawContours(mask, [contour[:, 0, ::-1]], -1, 255, cv.FILLED)
        mean_color = cv.mean(color_image, mask=mask)[:3]
        return tuple(int(x) for x in mean_color)
        
    def check_color_order(self,color_image,depth,points):
        """
    Checks the order of colors in the detected objects based on provided points and compares with predefined cube colors.

    This function identifies contours in the image, finds contours near specified points, extracts colors from these
    contours, and then verifies if these detected colors match the expected colors of the cubes. It logs a message if 
    there is a mismatch.

    Args:
        color_image (np.ndarray): An image array containing the color data of the scene.
        depth (np.ndarray): An image array containing the depth data of the scene.
        points (list of tuple): A list of (x, y) tuples indicating points where the color check should be performed.

    Returns:
        None: This function prints output directly and does not return any value.
        """
        detected_colors = []
        contours = self.detect_objects(color_image,depth)[0]
        for pt in points:
            contour = self.find_contour_near_point(contours,pt)
            color = self.get_color_from_contour(color_image,contour)
            detected_colors.append(color)
         
        cubes =[ 
                'Tall_Green',
                'Short_Yellow',
                'Tall_Orange',
                'Short_White',
                'Short_Green',
                'Tall_Yellow',
                'Short_Orange',
                'Tall_White'
                ]
        for key,colors in zip(cubes,detected_colors):
            if not self.is_same_color(self.cubes[key]['color'],colors):
                print('Not same color',key,'expected: ',self.cubes[key]['color'], 'got : ',colors)

    def is_same_color(self, recorded_color, known_color):
        """
        Determines if two colors are the same within a defined tolerance level.
        
        Args:
            recorded_color (tuple): The first color to compare.
            known_color (tuple): The second color to compare.
        
        Returns:
            bool: True if the colors are the same within the tolerance, False otherwise.
        """
        tolerance = self.get_cfg('color_tolerance')
        for c1, c2 in zip(recorded_color[:3], known_color):
            if abs(c1 - c2) > tolerance:
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
        
        return final_bin_im,min_depth

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
    T = np.array([
        [np.cos(phi),np.sin(phi)*np.sin(theta),np.sin(phi)*np.cos(theta),0],
        [0,np.cos(theta),-np.sin(theta),0],
        [-np.sin(phi),np.sin(theta)*np.cos(phi),np.cos(theta)*np.cos(phi),0],
        [0,0,0,1]
        ])
    # Find mean color of calibration cubes:
    pts =[(276,77),(332,92),(403,72),(455,95),(263,340),(351,330),(427,340),(506,347)]

    for _ in range(60): #delay to get accurate readings
        color,depth = camera.frames()
        depth = camera.transform_image(depth,T)
    detector.find_calibration_cubes(color,depth)
    detector.check_color_order(color,depth,pts)
    detector.draw_cube_points()
    #Main event loop:    
    while True:
        color, depth = camera.frames()
        depth = camera.transform_image(depth,T)
        depth_im = DepthImage(depth)
        contours,tops,color_mask,depth_mask = detector.detect_objects(color, depth)
        cv.imshow('color_mask',color_mask._image_data())
        cv.imshow('depth_mask',depth_mask._image_data())
        cv.imshow('tops',tops._image_data())
        cv.imshow("color", color)
        if runflag:
            # Get segmask of object nearest mouseclick
            runflag = False
            cv.waitKey(1)
                  
            containing_contour = detector.find_contour_near_point(contours,posList[0])
            print(detector.get_color_from_contour(color,containing_contour))
            posList.pop(0)
            if containing_contour is None:
                print("no contour found")
            else:
                single_obj_bin_im = tops.contour_mask(containing_contour)
                cv.imshow('result',single_obj_bin_im._image_data())
            
             
        cv.waitKey(1)
        
       
