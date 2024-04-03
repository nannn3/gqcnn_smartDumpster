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
    COLOR_IM_HEIGHT = 480
    COLOR_IM_WIDTH = 640
    DEPTH_IM_HEIGHT = 480
    DEPTH_IM_WIDTH = 640
    FOCAL_X = 525.0
    FOCAL_Y = 525.0
    FPS = 30

    def __init__(self,
            frame="Astra",
            depth_format=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
            color_format=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
            mirror_depth = True,
            mirror_color = True)
        self._depth_format = depth_format
        self._color_format = color_format
        self._mirror_depth = mirror_depth
        self._mirror_color = mirror_color
        self._device = None
        self._color_stream = None
        self._depth_stream = None

    def start(self):
        '''
        Initalize the sensor and create the streams
        '''
        openni2.initialize()
        try:
            self._device = openni2.Device.open_any()
            self._color_stream = self._device.create_depth_stream()
            self._depth_stream = self._device.create_color_stream()
            self._color_stream.set_mirroring_enabled(self._mirror_color)
            self._depth_stream.set_mirroring_enabled(self._mirror_depth)
        except Exception as ex:
            print("Error: Unable to open the device: ",ex)
            return
        
        self._color_stream.set_video_mode(pixelFormat=self._color_format,
                resolutionX=Astra.COLOR_IM_WIDTH,
                resolutionY=Astra.COLOR_IM_HEIGHT,
                fps=Astra.FPS)
        self._depth_stream.set_video_mode(pixelFormat=self._depth_format,
                resolutionX=Astra.DEPTH_IM_WIDTH,
                resolutionY=Astra.DEPTH_IM_HEIGHT,
                fps = Astra.FPS)
        self._device.set_image_registration_mode(True)
        self._device.set_depth_color_sync_enabled(True)
        self._depth_stream.start()
        self._color_stream.start()

