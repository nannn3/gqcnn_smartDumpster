import pdb
import scipy.ndimage as snm
import numpy as np
from autolab_core import ColorImage, DepthImage, BinaryImage, Contour
import cv2 as cv
import json
if __name__ == "__main__":
    import detectedObject
else:
    from . import detectedObject

class Detector:
    def __init__(self, config_file):
        """
        Initialize the Detector with a configuration file.

        Args:
            config_file (str): Path to the configuration file.
        """
        self.load_cfg(config_file)
        self.cubes = { # Expected calibration cubes and their average colors:
                'Tall_Green':{'color':(55,110,160)},
                'Short_Blue':{'color':(20,195,80)},
                'Tall_Orange':{'color':(100,120,235)},
                'Short_White':{'color':(5,5,225)},
                'Short_Green':{'color':(55,55,225)},
                'Tall_Blue':{'color':(10,190,145)},
                'Short_Orange':{'color':(100,115,245)},
                'Tall_White':{'color':(5,5,245)}
                }
        self.detected_objects = {}

        for name,prop in self.cubes.items():
            # Update to DetectedObject class
            cube = detectedObject.DetectedObject(name,color = prop['color'])
            self.cubes[name] = cube

    def draw_cube_points(self,color_image):
        color_image = color_image.copy()
        for name,detected in self.cubes.items():
            try:
                contour = detected.get_property('contour')
                box = contour.bounding_box
                y,x = box.center
                x = int(x)
                y = int(y)
                contour = contour.boundary_pixels.reshape(-1,1,2).astype(np.int32)
                contour = contour[:, 0, ::-1] #Swap x, y coordinates
                color_image = cv.drawContours(color_image,[contour],-1,(255,0,0),thickness = 2)
                color_image = cv.putText(color_image,name,(x,y),cv.FONT_HERSHEY_PLAIN,1,(255,0,0))
            except KeyError:
                print(f"{name} has no contour")

        cv.imshow('calibration_cubes',color_image)

    def compare_color_to_cubes(self, color):
        """
        Compares a given color to predefined cube colors and returns the matching cube name.
        
        Args:
            color (tuple): The color to compare.
        
        Returns:
            str: Name of the cube if a match is found, otherwise None.
        """
        for cube_name, cube in self.cubes.items():
            if cube.is_same_color(color):
                return cube_name
        return None
    
    def find_calibration_cubes(self, color_image, depth):
        """
        Finds all calibration cubes and categorizes them by color and height.
        It specifically differentiates between two white cubes based on their relative heights.

        Args:
            color_image (np.ndarray): The color image.
            depth (np.ndarray): The depth image.

        Raises:
            ValueError: If there are fewer contours detected than cubes expected in self.cubes.
        """
        contours = self.detect_objects(color_image, depth)[0]
        if len(contours) < len(self.cubes):
            raise ValueError("There should be at least as many contours as cubes")

        white_cubes = []
        color_image = cv.cvtColor(color_image, cv.COLOR_RGB2HSV)
        for contour in contours:
            color = self.get_color_from_contour(color_image, contour)
            cube_name = self.compare_color_to_cubes(color)
            if cube_name is None:
                continue

            height = self.find_object_tops(depth, [contour])[1]
            cube_prop = {'contour': contour, 'height': height}

            # Collect height data for white cubes to compare later
            if 'White' in cube_name:
                white_cubes.append((cube_name, cube_prop))
            else:
                self.cubes[cube_name].update_properties(**cube_prop)

        # Compare white cubes based on height and assign the correct labels
        if white_cubes:
            # Sort by height descending; higher height means further away
            sorted_white_cubes = sorted(white_cubes, key=lambda x: x[1]['height'], reverse=True)
            try:
                self.cubes['Short_White'].update_properties(**sorted_white_cubes[0][1])
                self.cubes['Tall_White'].update_properties(**sorted_white_cubes[1][1])
            except IndexError:
                pass


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
        hsv = cv.cvtColor(color_image,cv.COLOR_RGB2HSV)
        for pt in points:
            contour = self.find_contour_near_point(contours,pt)
            color = self.get_color_from_contour(hsv,contour)
            detected_colors.append(color)
         
        cubes =[ 
                'Tall_Green',
                'Short_Blue',
                'Tall_Orange',
                'Short_White',
                'Short_Green',
                'Tall_Blue',
                'Short_Orange',
                'Tall_White'
                ]
        for key,colors in zip(cubes,detected_colors):
            cube = self.cubes[key]
            if not cube.is_same_color(colors):
                print('Not same color',key,'expected: ',self.cubes[key].color, 'got : ',colors)


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
    pts =[(245,72),(305,90),(370,64),(423,82),(261,344),(340,324),(411,324),(497,298)]

    for _ in range(60): #delay to get accurate readings
        color,depth = camera.frames()
        depth = camera.transform_image(depth,T)
    detector.find_calibration_cubes(color,depth)
    detector.check_color_order(color,depth,pts)
    detector.draw_cube_points(color)
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
            color = cv.cvtColor(color,cv.COLOR_RGB2HSV)
            print(detector.get_color_from_contour(color,containing_contour))
            posList.pop(0)
            if containing_contour is None:
                print("no contour found")
            else:
                single_obj_bin_im = tops.contour_mask(containing_contour)
                cv.imshow('result',single_obj_bin_im._image_data())
            
             
        cv.waitKey(1)
        
       
