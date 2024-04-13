import cv2
from openni import openni2
from openni import _openni2 as c_api
import numpy as np
from autolab_core import CameraIntrinsics
import os

METERS_TO_MM = 1000.0
MM_TO_METERS = 1.0 / METERS_TO_MM

# Path to the OpenNI redistribution directory
OPENNI_REDIST = os.environ.get('OPENNI2_REDIST',None)

if OPENNI_REDIST is None:
    raise ImportError("OPENNI2_REDIST enviornment vairable not set. Make sure that openni2 is installed")


class Astra():
    """Class representing the Astra camera sensor."""
    # Default settings for the Astra camera
    WIDTH = 640
    HEIGHT = 480
    FPS = 30
    MIRRORING = True

    def __init__(self,color_intr=None,ir_intr=None):
        """Initialize Astra camera.
        Param:
            color_intr file path to .intr file containing data for the color camera
            ir_intr file path to .intr file containing data for the ir camera
        """
        self._color_stream = None
        self._depth_stream = None
        self._dev = None
        self._color_intr = None
        if color_intr is not None:
            self._color_intr = CameraIntrinsics.load(color_intr)
        self._ir_intr = None
        if ir_intr is not None:
            self._ir_intr = CameraIntrinsics.load(ir_intr)
        self._running = False
        
    def __del__(self):
        if self.is_running:
            self.stop()
    
    @property
    def is_running(self):
        return self._running
    
    @property
    def  ir_intrinsics(self):
        '''
        Return: 
            CameraIntrinsics obj
        '''
        return self._ir_intr
    
    @property
    def color_intrinsics(self):
        '''
        Return:
            CameraIntrinsics obj
        '''
        return self._color_intr
        
    def start(self):
        """Start the Astra camera."""
        # Initialize OpenNI
        openni2.initialize(OPENNI_REDIST)
        # Open any available device
        self._dev = openni2.Device.open_any()
        # Create depth and color streams
        self._depth_stream = self._dev.create_depth_stream()
        self._color_stream = self._dev.create_color_stream()
        # Set mirroring for both streams
        self._depth_stream.set_mirroring_enabled(self.MIRRORING)
        self._color_stream.set_mirroring_enabled(self.MIRRORING)
        
        # Configure depth and color stream video modes
        #TODO is this actually required?
        self._depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                                       resolutionX=self.WIDTH,
                                                       resolutionY=self.HEIGHT,
                                                       fps=self.FPS))
        self._color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                                       resolutionX=self.WIDTH,
                                                       resolutionY=self.HEIGHT,
                                                       fps=self.FPS))
        
        # Enable depth-color synchronization and image registration to allow the camera to autmatically deliver frames at the same time
        self._dev.set_image_registration_mode(True)
        self._dev.set_depth_color_sync_enabled(True)
        # Start depth and color streams
        self._depth_stream.start()
        self._color_stream.start()
        self._running = True

    def frames(self):
        """Read frames from the Astra camera.
            Returns:
                list of np.NDArrays representing the color as a 480x640x3 png and depth as a 480x640x1 array of floats representing the distance in Meters
        """
        # Read color and depth frames
        frame_color = self._color_stream.read_frame()
        frame_depth = self._depth_stream.read_frame()
        frame_color_data = frame_color.get_buffer_as_uint8() #if this is not a uint8, numpy won't be able to seperate it correctly
        frame_depth_data = frame_depth.get_buffer_as_uint16()
        
        # Convert color frame data to numpy array
        color_array = np.ndarray((frame_color.height, frame_color.width, 3),dtype=np.uint8,buffer=frame_color_data)
        color_array = cv2.cvtColor(color_array, cv2.COLOR_BGR2RGB) # The camera give it to us in BGR. We need to switch it to RGB to be useful to us.
        
        # Convert depth frame data to numpy array
        depth_array = np.ndarray((frame_depth.height, frame_depth.width),dtype=np.uint16,buffer=frame_depth_data)
        depth_array = depth_array * MM_TO_METERS #covert to a float array in meters 
        return [color_array, depth_array]
    
    def stop(self):
        """Stop the Astra camera."""
        # Stop depth and color streams
        self._depth_stream.stop()
        self._color_stream.stop()
        self._running = False

if __name__ == '__main__':
    '''
    Quick test harness. Should display live feed for both depth and color
    '''
    camera = Astra()
    camera.start()
    while 1:
        # Read frames from the camera
        color,depth = camera.frames()
        # Display depth and color frames
        cv2.imshow('depth',depth)
        cv2.imshow('color',color)
        cv2.waitKey(1)
