import pdb
import os
import json
import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
import open3d as o3d
from autolab_core import CameraIntrinsics, PointCloud, DepthImage
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

    def __init__(self, color_intr_path=None, ir_intr_path=None):
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
            return None

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
        '''       
        depth = DepthImage(depth_array,frame = self.color_intrinsics.frame)
        return self.color_intrinsics.deproject(depth)
        '''
    def point_cloud_to_depth(self,point_cloud):
        depth_image = np.zeros((480,640))
        data = point_cloud.data
        data = data.reshape(-1,3)
        for point in data:
            x,y,z = point
            if z == 0:
                continue
            u = int((x * self.color_intrinsics.fx / z) + self.color_intrinsics.cx)
            v = int((y * self.color_intrinsics.fy / z) +  self.color_intrinsics.cy)
            if 0 <= u <= 640 and 0<=v <480:
                depth_image[v,u]=z
        return depth_image

    def stop(self):
        """
        Stops the depth and color streams and unloads the OpenNI environment.
        """
        if self._depth_stream and self._color_stream:
            self._depth_stream.stop()
            self._color_stream.stop()
            self._running = False
            openni2.unload()

def depth_to_color(depth, max_depth):
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

def visualize_point_cloud(point_cloud):
    pcd = o3d.geometry.PointCloud()
    data = point_cloud.data.reshape(-1,3)
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    from PIL import Image
    counter = 1
    file_path = os.path.dirname(os.path.realpath(__file__))
    color_intr_file = os.path.join(file_path,'Astra_Color.intr')
    ir_intr_file = os.path.join('file_path,Astra_IR.intr')
    # Initialize the camera
    camera = Astra(color_intr_path = color_intr_file, ir_intr_path = ir_intr_file)
    camera.start()
    color_intr = camera.color_intrinsics


    try:
        while True:
            # Get color and depth frames from the camera
            color, depth = camera.frames()

            # Convert the depth data to a point cloud (if needed for processing, not visualization)
            point_cloud = camera.depth_to_point_cloud(depth)

            # Convert depth to a visual format
            max_depth = np.max(depth) if np.max(depth) > 0 else 1.0  # Prevent division by zero
            depth_display = depth_to_color(depth, max_depth)

            # Display the color and depth images
            cv2.imshow('Color', color)
            cv2.imshow('Depth', depth_display)
            key = cv2.waitKey(1) & 0xFF 
            if key == ord('q'):
                break
            elif key ==ord('p'):
                visualize_point_cloud(point_cloud)
                np.savetxt('foo.csv',point_cloud.data.reshape(-1,3))
            
            elif key == ord('c'):
                im = Image.fromarray(color)
                image_path = os.path.join(file_path,'..','Calibration_Pics')
                im.save(f'image{counter}.png')
            
            elif key == ord('v'):
                depth_from_point = camera.point_cloud_to_depth(point_cloud)
                cv2.imshow('depth from point cloud',depth_from_point)
                cv2.waitKey(1)

    finally:
        camera.stop()
        cv2.destroyAllWindows()
