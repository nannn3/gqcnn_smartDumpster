import pdb
import os
import json
import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
from autolab_core import CameraIntrinsics, DepthImage

#Default intr paths:
color_intr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Astra_Color.intr')
ir_intr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Astra_IR.intr')

class Astra():
    """
    Class representing the Astra camera sensor with configurable color and IR intrinsics.
    Manages both color and depth streams.
    """

    WIDTH = 640
    HEIGHT = 480
    FPS = 30
    DEPTH_MIRRORING = True
    COLOR_MIRRORING = True 

    def __init__(self, color_intr_path=color_intr_path, ir_intr_path=ir_intr_path):
        """
        Initializes the Astra camera with paths to intrinsic files for the color and IR cameras.

        Args:
            color_intr_path (str): File path to the .intr file containing data for the color camera.
            ir_intr_path (str): File path to the .intr file containing data for the IR camera.
        """
        self._color_stream = None
        self._depth_stream = None
        self._dev = None
        self._running = False
        self._color_intr = self.load_intrinsics(color_intr_path)
        self._ir_intr = self.load_intrinsics(ir_intr_path)

    def __del__(self):
        """Ensures that the camera streams are stopped when the instance is destroyed."""
        if self.is_running:
            self.stop()
    
	def compute_best_rotation_matrix(self, depth, points):
        """
        Computes the best rotation matrix based on the given depth image and a list of points.
        The function iteratively refines the range and step size to find the optimal theta and phi.

        Args:
            depth (numpy.ndarray): The depth image array to transform.
            points (list of tuples): A list of (row, col) tuples representing points in the depth image.

        Returns:
            numpy.ndarray: The best 4x4 rotation matrix.
        """
        def adjust_range_and_step(value, range_min, range_max, step_size):
            """
            Adjusts the range and step size based on the current best value.

            Args:
                value (float): The current best value.
                range_min (float): The minimum value of the range.
                range_max (float): The maximum value of the range.
                step_size (float): The current step size.

            Returns:
                tuple: A tuple containing the new range_min, range_max, and step_size.
            """
            if (value - range_min < step_size) or (range_max - value < step_size):
                # Near the bounds, shif the step size
                step_size /= 10

            range_min = value - 5 * step_size
            range_max = value + 5 * step_size
            return range_min, range_max, step_size

        min_dif = np.inf
        best_theta = 0
        best_phi = 0

        theta_min, theta_max, theta_step = -1, 0, 0.1
        phi_min, phi_max, phi_step = -1, 0, 0.1

        while theta_step > 1e-4 or phi_step > 1e-4:
            for theta_deg in np.arange(theta_min, theta_max, theta_step):
                theta = theta_deg * (np.pi / 180)
                for phi_deg in np.arange(phi_min, phi_max, phi_step):
                    phi = phi_deg * (np.pi / 180)

                    T = np.array([
                        [np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi), 0],
                        [0, np.cos(theta), -np.sin(theta), 0],
                        [-np.sin(phi), np.sin(theta) * np.cos(phi), np.cos(theta) * np.cos(phi), 0],
                        [0, 0, 0, 1]
                    ])

                    transformed_depth = self.transform_image(depth, T)
                    vals = [transformed_depth[row, col] for row, col in points]

                    if 0 in vals:
                        continue

                    dif = average_dif(vals)

                    if dif < min_dif:
                        best_theta = theta_deg
                        best_phi = phi_deg
                        min_dif = dif

            theta_min, theta_max, theta_step = adjust_range_and_step(best_theta, theta_min, theta_max, theta_step)
            phi_min, phi_max, phi_step = adjust_range_and_step(best_phi, phi_min, phi_max, phi_step)

            # Check stopping condition
            if min_dif < 0.001:
                break

        theta = best_theta * (np.pi / 180)
        phi = best_phi * (np.pi / 180)

        R = np.array([
            [np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta) * np.sin(phi), 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [-np.sin(phi), np.sin(theta) * np.cos(phi), np.cos(theta) * np.cos(phi), 0],
            [0, 0, 0, 1]
        ])
        return R

    @staticmethod
    def load_intrinsics(path):
        """
        Loads camera intrinsics from a specified JSON file path.

        Args:
            path (str): Path to the JSON file containing camera intrinsics data.

        Returns:
            CameraIntrinsics: An object containing the camera intrinsics if the file exists, None otherwise.
        """
        if path and os.path.exists(path):
            return CameraIntrinsics.load(path)

        else:
            raise ValueError("Cannot open intrinsics file:",path)

    @property
    def is_running(self):
        """Checks if the camera streams are currently active."""
        return self._running

    @property
    def ir_intrinsics(self):
        """Returns the IR camera intrinsics."""
        return self._ir_intr
    
    @property
    def color_intrinsics(self):
        """Returns the color camera intrinsics."""
        return self._color_intr

    def start(self):
        """
        Initializes and starts the depth and color streams of the Astra camera.
        Raises an exception if initialization fails.
        """
        try:
            openni2.initialize(os.environ.get('OPENNI2_REDIST', None))
            self._dev = openni2.Device.open_any()
        except Exception as e:
            print(f"Failed to initialize the Astra camera: {str(e)}")
            raise

        self._depth_stream = self._dev.create_depth_stream()
        self._color_stream = self._dev.create_color_stream()
        self.configure_streams()
        self._depth_stream.start()
        self._color_stream.start()
        self._color_stream.set_property(c_api.ONI_STREAM_PROPERTY_GAIN,550)
        self._running = True

    def configure_streams(self):
        """
        Configures the video modes for both the depth and color streams.
        Also enables depth-color synchronization.
        """
        self._depth_stream.set_video_mode(c_api.OniVideoMode(
            pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
            resolutionX=self.WIDTH, resolutionY=self.HEIGHT, fps=self.FPS))
        self._color_stream.set_video_mode(c_api.OniVideoMode(
            pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
            resolutionX=self.WIDTH, resolutionY=self.HEIGHT, fps=self.FPS))
        self._color_stream.set_property(c_api.ONI_STREAM_PROPERTY_AUTO_WHITE_BALANCE,False)
        self._color_stream.set_property(c_api.ONI_STREAM_PROPERTY_AUTO_EXPOSURE,False)
        self._dev.set_image_registration_mode(True)
        self._dev.set_depth_color_sync_enabled(True)

    def frames(self):
        """
        Reads frames from the Astra camera.

        Returns:
            list: A list containing two numpy arrays, one representing the color frame and the other the depth frame.
        """
        frame_color = self.read_color_frame()
        frame_depth = self.read_depth_frame()
        return [frame_color, frame_depth]

    def read_color_frame(self):
        """
        Reads and processes the color frame from the camera.

        Returns:
            numpy.ndarray: The processed color image array.
        """
        frame = self._color_stream.read_frame()
        data = frame.get_buffer_as_uint8()
        array = np.ndarray((frame.height, frame.width, 3), dtype=np.uint8, buffer=data)
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        if self.COLOR_MIRRORING:
            array = np.fliplr(array)
        return array

    def read_depth_frame(self):
        """
        Reads and processes the depth frame from the camera.

        Returns:
            numpy.ndarray: The processed depth image array.
        """
        frame = self._depth_stream.read_frame()
        data = frame.get_buffer_as_uint16()
        array = np.ndarray((frame.height, frame.width), dtype=np.uint16, buffer=data)
        array = self.preprocess_depth_array(array)
        if self.DEPTH_MIRRORING:
            array = np.fliplr(array)
        return array

    def preprocess_depth_array(self, depth):
        """
        Applies preprocessing to the depth data, converting from millimeters to meters and applying a median blur filter.

        Args:
            depth (np.ndarray): The depth image array in millimeters.

        Returns:
            np.ndarray: The preprocessed depth image array in meters.
        """
        depth = depth.astype('float32')
        depth = cv2.medianBlur(depth, 3)
        depth *= (1.0 / 1000.0)  # Convert mm to meters
        depth[depth >=1] = 0 #Filter out things further than 1.25 m
        
        #TODO Move this to its own function that takes a range of values to remove dead space in testing env
        for row in range(380,460,1):
            for col in range(46,257,1):
                depth[row][col] = 0
        return depth

    def transform_image(self, image, T,threshold = .4):
        """
        Applies a 4x4 transformation matrix T to the depth image.
        
        Args:
            image (numpy.ndarray): The depth image array to transform.
            T (numpy.ndarray): A 4x4 transformation matrix.
            threshold (float): The number below which all values are set to 0
        
        Returns:
            numpy.ndarray: The transformed depth image array.
        
        Raises:
            ValueError: If T is not a 4x4 matrix.
        """
        if T.shape != (4, 4):
            raise ValueError("Transformation matrix T must be a 4x4 matrix")

        # Generate grid of coordinates (x, y, z, 1)
        coord_x, coord_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        coord_z = image[coord_y, coord_x]
        ones = np.ones_like(coord_z)
        coords = np.stack([coord_x, coord_y, coord_z, ones], axis=-1)

        # Perform matrix multiplication
        transformed_coords = coords.dot(T)  

        # Extract the z-coordinate after transformation
        transformed_image = transformed_coords[..., 2]
        transformed_image[transformed_image < threshold] = 0
        return transformed_image.reshape(image.shape)
       
    def stop(self):
        """
        Stops the depth and color streams and unloads the OpenNI environment.
        """
        if self._depth_stream and self._color_stream:
            self._depth_stream.stop()
            self._color_stream.stop()
            self._running = False
            openni2.unload()

    def depth_to_color(self,depth):
        """
        Normalize the depth image and convert to a colormap for visualization.
        
        Args:
            depth (numpy.ndarray): The depth image where values represent depth in meters.
            max_depth (float): The maximum depth value to normalize.

        Returns:
            numpy.ndarray: A colorized depth image.
        """
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        return depth_colored

'''
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
		MODULE FUNCTIONS FOR TESTING
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

def visualize_point_cloud(point_cloud):
    #removed in git commit 3d992170d85b8dfdc344f5c46b14264929fc3dfd
    raise NotImplemented('Removed in commit 3d992170d85b8dfdc344f5c46b14264929fc3dfd')

def rotate_and_visualize_point_cloud(camera,depth, R):
    #removed in git commit 3d992170d85b8dfdc344f5c46b14264929fc3dfd
    raise NotImplemented('Removed in commit 3d992170d85b8dfdc344f5c46b14264929fc3dfd')

def save_color_image(color, counter):
    """
    Saves the current color frame from the camera as an image file.

    Args:
        camera (Astra): The camera instance.
        counter (int): Image counter used for naming the saved file.
    """
    im = Image.fromarray(color)
    image_path = os.path.join(os.path.dirname(__file__), '..', 'Calibration_Pics')
    im.save(os.path.join(image_path,f'image{counter}.png'))

def view_depth_from_point_cloud(camera,depth, R):
    #removed in commit 3d992170d85b8dfdc344f5c46b14264929fc3dfd
    raise NotImplemented("Removed in commit 3d992170d85b8dfdc344f5c46b14264929fc3dfd")

def process_user_input(camera, T, depth,color):
    """
    Processes user input to control camera operations and visualization.

    Args:
        camera (Astra): The camera instance.
        R (numpy.ndarray): Rotation matrix.
        counter (int): Image counter for naming saved images.

    Returns:
        bool: False if the user pressed 'q' to quit, True otherwise.
    """
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        return False  # Signal to break the loop and exit
    elif key == ord('p'):
        point_cloud = camera.depth_to_point_cloud(depth)
        visualize_point_cloud(point_cloud)
    elif key == ord('r'):
        transformed_depth = camera.transform_image(depth,T)
        min_depth = np.min(transformed_depth[np.nonzero(transformed_depth)])
        i, j = np.where(np.isclose(transformed_depth, min_depth)) 
        print("Pixel with highest depth value is:",(i,j))
        depth = camera.depth_to_color(transformed_depth)
        
        cv2.imshow('Transformed image',depth)
    elif key == ord('c'):
        global counter
        save_color_image(color, counter)

        counter += 1
    elif key == ord('v'):
        view_depth_from_point_cloud(camera,depth, R)
    return True

def average_dif(vals):
    '''
    calculates the average distance between values
    Args:
        vals: list of floats
    returns float: average of the differences
    '''
    n = len(vals)
    total_dif = 0
    count = 0
    for i in range(n):
        for j in range(i+1,n):
            total_dif += abs(vals[i]-vals[j])
            count += 1
    return total_dif/count

counter = 1
if __name__ == "__main__":
    """
    Initializes the camera, captures frames, and processes user input until termination.
    """
    from PIL import Image 
    camera = Astra(color_intr_path,ir_intr_path)
    camera.start()
    color,depth = camera.frames() 
    
    # TEST to find best theta value
    min_dif = np.inf
    best_theta = 0;
    best_phi = 0
    for i in range(-100,100,2):
        for j in range(-100,100,2)
        theta_deg = i*.001
        theta = theta_deg*(np.pi/180)
        phi_deg = j * .001
        phi = phi_deg *(np.pi/180)

        T = np.array([
            [np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)*np.sin(phi),0],
            [0,np.cos(theta),-np.sin(theta),0],
            [-np.sin(phi),np.sin(theta)*np.cos(phi),np.cos(theta)*np.cos(phi),0],
            [0,0,0,1]
            ])
        transformed_depth = camera.transform_image(depth,T)
        
        d1 = transformed_depth[45][272]
        d2 = transformed_depth[46][382]
        d3 = transformed_depth[311][345]
        d4 = transformed_depth[299][509]
        d5 = transformed_depth[164][370]
        vals = [d1,d2,d3,d4,d5]
        if 0 in vals:
            continue
        dif = average_dif(vals)

        if dif < min_dif:
            best_theta = theta_deg
            best_phi = phi_deg
            min_dif = dif
    print('Best found theta = ',best_theta,'Best Phi = ',best_phi,'Best average dif:',min_dif)
    ### end test
    theta =best_theta
    theta *= (np.pi/180)
    phi = best_phi
    phi *= (np.pi/180)

    T = np.array([
        [np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)*np.sin(phi),0],
        [0,np.cos(theta),-np.sin(theta),0],
        [-np.sin(phi),np.sin(theta)*np.cos(phi),np.cos(theta)*np.cos(phi),0],
        [0,0,0,1]
        ])
    color,depth = camera.frames()
    #for i in range(250,500,50):
     #   camera._color_stream.set_property(c_api.ONI_STREAM_PROPERTY_GAIN,i)
      #  color,depth = camera.frames()
       # cv2.imshow(str(i),color)
    try:
        while True:
            color, depth = camera.frames()
            # Crop to active work zone:
            #color = color[20:250,115:500]
            #depth = depth[20:250,115:500]
            depth = camera.transform_image(depth,T)
            max_depth = np.max(depth)
            min_depth = np.min(depth)
            print(max_depth,min_depth)
            depth_display = camera.depth_to_color(depth)
            cv2.imshow('Color', color)
            cv2.imshow('Depth', depth_display)
            if not process_user_input(camera, T,depth,color):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


