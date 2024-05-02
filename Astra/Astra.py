import pdb
import os
import json
import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import open3d as o3d
from autolab_core import CameraIntrinsics, PointCloud, DepthImage

#Default intr paths:
color_intr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Astra_Color.intr')
ir_intr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'Astra_IR.intr')

class Astra():
    """
    Class representing the Astra camera sensor with configurable color and IR intrinsics.
    Manages both color and depth streams and allows the generation of point clouds from depth data.
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

    def depth_to_point_cloud(self, depth_array):
        """
        Converts a depth array into a 3D point cloud based on the intrinsic parameters of the camera.

        Args:
            depth_array (numpy.ndarray): A depth image where each pixel represents depth in meters.

        Returns:
            numpy.ndarray: A 3D point cloud represented as a three-dimensional array (X, Y, Z coordinates).
        """
        
        height, width = depth_array.shape
        x_indices = np.arange(width) - self.color_intrinsics.cx
        y_indices = np.arange(height) - self.color_intrinsics.cy
        x_indices, y_indices = np.meshgrid(x_indices, y_indices)

        Z = depth_array
        X = np.multiply(x_indices, Z) / self.color_intrinsics.fx
        Y = np.multiply(y_indices, Z) / self.color_intrinsics.fy
        data = np.dstack((X, Y, Z))  # Stack along the third dimension
        
        data = data.reshape(3,-1)
        return PointCloud(data,frame = self.color_intrinsics.frame)
    
    def point_cloud_to_depth(self,point_cloud):
        #TODO figure out why this doesn't work the way it should
        depth_image = np.zeros((Astra.HEIGHT,Astra.WIDTH))
        data = point_cloud.data
        data = data.reshape(-1,3)
        for point in data:
            x,y,z = point
            if z == 0:
                continue
            u = int((x * self.color_intrinsics.fx / z) + self.color_intrinsics.cx)
            v = int((y * self.color_intrinsics.fy / z) +  self.color_intrinsics.cy)
            if 0 <= u < Astra.WIDTH and 0<=v < Astra.HEIGHT:
                depth_image[v,u]=abs(z) #If there's z<0, that just means its under the camera. This is expected

        depth_image =  np.flipud(depth_image)
        T = np.float32([[1,0,0],[0,1,-50]]) #Translation matrix to shift up by 50 px
        translated_depth = cv2.warpAffine(depth_image,T,(Astra.WIDTH,Astra.HEIGHT))
        translated_depth[-50:] = 0 #Replace bottom 50 rows with 0s
        return translated_depth

    def rotate_point_cloud(self,point_cloud,R):
        '''Applies Rotation matrix R to a point cloud
        returns new point cloud that has been rotated
        '''
        rotated =point_cloud.data.reshape(-1,3).dot(R)
        point_cloud = PointCloud(rotated.reshape(3,-1),point_cloud.frame)
        return point_cloud

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
    """
    Visualizes a 3D point cloud using Open3D.

    Args:
        point_cloud (PointCloud): The point cloud to visualize.
    """
    pcd = o3d.geometry.PointCloud()
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    data = point_cloud.data.reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(data)
    viewer.add_geometry(pcd)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()

def rotate_and_visualize_point_cloud(camera,depth, R):
    """
    Rotates and visualizes a point cloud generated from the depth data of the camera.

    Args:
        camera (Astra): The camera instance.
        R (numpy.ndarray): Rotation matrix.
    """
    point_cloud = camera.depth_to_point_cloud(depth)
    point_cloud = camera.rotate_point_cloud(point_cloud, R)
    visualize_point_cloud(point_cloud)

def save_color_image(color, counter):
    """
    Saves the current color frame from the camera as an image file.

    Args:
        camera (Astra): The camera instance.
        counter (int): Image counter used for naming the saved file.
    """
    im = Image.fromarray(color)
    image_path = os.path.join(os.path.dirname(__file__), '..', 'Calibration_Pics')
    im.save(f'image{counter}.png')

def view_depth_from_point_cloud(camera,depth, R):
    """
    Converts a rotated point cloud back to depth data and displays it.

    Args:
        camera (Astra): The camera instance.
        R (numpy.ndarray): Rotation matrix.
    """
    point_cloud = camera.depth_to_point_cloud(depth)
    point_cloud = camera.rotate_point_cloud(point_cloud, R)
    depth_from_point = camera.point_cloud_to_depth(point_cloud)
    depth_display = camera.depth_to_color(depth_from_point)
    cv2.imshow('Depth from Point Cloud', depth_display)

def process_user_input(camera, R, counter,depth,color):
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
        rotate_and_visualize_point_cloud(camera,depth, R)
    elif key == ord('c'):
        save_color_image(color, counter)
    elif key == ord('v'):
        view_depth_from_point_cloud(camera,depth, R)
    return True

if __name__ == "__main__":
    """
    Initializes the camera, captures frames, and processes user input until termination.
    """
    camera = Astra(color_intr_path,ir_intr_path)
    camera.start()
    counter = 1
    
    R = np.array([
        [-.9998, .0095, -.0196],
        [.0112, .9957, -.0924],
        [.0187, -.0926, -.9955]
    ]) # Example Rotation Matrix

    try:
        while True:
            color, depth = camera.frames() 
            depth_display = camera.depth_to_color(depth)
            cv2.imshow('Color', color)
            cv2.imshow('Depth', depth_display)
            if not process_user_input(camera, R, counter,depth,color):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()


