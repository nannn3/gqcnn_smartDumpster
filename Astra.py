import cv2 as cv
import numpy as np
from openni import openni2
from openni import _openni2 as c_api
from perception import CameraSensor
from autolab_core import CameraIntrinsics

class Astra(CameraSensor):
    '''
    Class for interacting with the Orbbecc Astra RGB+D camera
    '''
    # Constants for image dimensions and camera parameters
    COLOR_IM_HEIGHT = 480
    COLOR_IM_WIDTH = 640
    DEPTH_IM_HEIGHT = 480
    DEPTH_IM_WIDTH = 640
    FOCAL_X = 525.0
    FOCAL_Y = 525.0
    FPS = 30

    def __init__(self,
            frame="Astra",
            mirror_depth=True,
            mirror_color=True):
        '''
        Constructor for Astra camera sensor
        
        Args:
            frame (str): Name of the camera frame.
            depth_format: Pixel format for the depth stream.
            color_format: Pixel format for the color stream.
            mirror_depth (bool): Whether to mirror the depth stream.
            mirror_color (bool): Whether to mirror the color stream.
        '''
        self._mirror_depth = mirror_depth
        self._mirror_color = mirror_color
        self._device = None
        self._color_stream = None
        self._depth_stream = None

    def start(self):
        '''
        Initialize the sensor and create the streams
        '''
        openni2.initialize()
        try:
            # Attempt to open the device
            self._device = openni2.Device.open_any()
            # Create depth and color streams
            self._color_stream = self._device.create_depth_stream()
            self._depth_stream = self._device.create_color_stream()

        except Exception as ex:
            print("Error: Unable to open the device: ", ex)
            return
        # Set mirroring for depth and color streams
        self._color_stream.set_mirroring_enabled(self._mirror_color)
        self._depth_stream.set_mirroring_enabled(self._mirror_depth)
        
        # Configure video modes for color and depth streams
        self._color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                resolutionX=Astra.COLOR_IM_WIDTH,
                resolutionY=Astra.COLOR_IM_HEIGHT,
                fps=Astra.FPS))
        self._depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                resolutionX=Astra.DEPTH_IM_WIDTH,
                resolutionY=Astra.DEPTH_IM_HEIGHT,
                fps = Astra.FPS)
        
        # Set image registration and depth-color sync
        self._device.set_image_registration_mode(True)
        self._device.set_depth_color_sync_enabled(True)
        
        # Start depth and color streams
        self._depth_stream.start()
        self._color_stream.start()
    
    def stop(self):
        '''Stops the sensor stream'''
        self._color_stream.stop()
        self._depth_stream.stop()

    def frames(self):
        '''
        Returns the color and depth frames as a list
        
        Returns:
            list: A list containing the color and depth frames.
        '''
        # Read color and depth frames from streams
        color_frame = self._color_stream.read_frame()
        depth_frame = self._depth_stream.read_frame()
        
        # Get data buffers for color and depth frames
        color_frame_data = color_frame.get_buffer_as_uint8()
        depth_frame_data = depth_frame.get_buffer_as_uint16()
        
        # Convert color frame data to numpy array and change color space from BGR to RGB
        color_array = np.ndarray((color_frame.height, color_frame.width, 3), dtype=np.uint8, buffer=color_frame_data)
        color_array = cv.cvtColor(color_array, cv.COLOR_BGR2RGB)
        
        # Convert depth frame data to numpy array
        depth_array = np.ndarray((depth_frame.height, depth_frame.width), dtype=np.uint16, buffer=depth_frame_data)
        
        # Return color and depth frames as a list
        return [color_array, depth_array]

if __name__ == "__main__":
    '''Quick test harness that should display the feeds'''
    camera = Astra()
    Astra.start()
    while 1:
        color,depth = Astra.frames()
        cv2.imshow('depth',depth)
        cv2.imshow('color',color)
        cv2.waitKey(1)