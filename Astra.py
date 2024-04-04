import cv2
from openni import openni2
from openni import _openni2 as c_api
import numpy as np
from perception import CameraSensor

class Astra(CameraSensor):
    # Constants moved inside the class
    WIDTH = 640
    HEIGHT = 480
    FPS = 30
    MIRRORING = True

    def __init__(self):
        self._color_stream = None
        self._depth_stream = None
        self._dev = None

    def start(self):
        openni2.initialize()
        self._dev = openni2.Device.open_any()
        self._depth_stream = self._dev.create_depth_stream()
        self._color_stream = self._dev.create_color_stream()
        self._depth_stream.set_mirroring_enabled(self.MIRRORING)
        self._color_stream.set_mirroring_enabled(self.MIRRORING)
        
        self._depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM,
                                                       resolutionX=self.WIDTH,
                                                       resolutionY=self.HEIGHT,
                                                       fps=self.FPS))
        self._color_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888,
                                                       resolutionX=self.WIDTH,
                                                       resolutionY=self.HEIGHT,
                                                       fps=self.FPS))
        
        self._dev.set_image_registration_mode(True)
        self._dev.set_depth_color_sync_enabled(True)
        self._depth_stream.start()
        self._color_stream.start()

    def frames(self):
        frame_color = self._color_stream.read_frame()
        frame_depth = self._depth_stream.read_frame()

        frame_color_data = frame_color.get_buffer_as_uint8()
        frame_depth_data = frame_depth.get_buffer_as_uint16()
        
        color_array = np.ndarray((frame_color.height, frame_color.width, 3),dtype=np.uint8,buffer=frame_color_data)
        color_array = cv2.cvtColor(color_array, cv2.COLOR_BGR2RGB)
        
        depth_array = np.ndarray((frame_depth.height, frame_depth.width),dtype=np.uint16,buffer=frame_depth_data)
        
        return [color_array, depth_array]
    
    def stop(self):
        self._depth_stream.stop()
        self._color_stream.stop()

if __name__ == '__main__':
    camera = Astra()
    camera.start()
    while 1:
        color,depth = camera.frames()
        cv2.imshow('depth',depth)
        cv2.imshow('color',color)
        cv2.waitKey(1)
