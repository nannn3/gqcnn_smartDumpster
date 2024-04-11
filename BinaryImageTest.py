from Astra import Astra
import cv2 as cv
from autolab_core import ColorImage, DepthImage, BinaryImage

if __name__ == "__main__":
    camera = Astra()
    camera.start()
    while (1):
        color,depth = camera.frames()
        color_im = ColorImage(color)
        bin_im = color_im.foreground_mask(50,ignore_black =False )
        cv.imshow("color",color)
        cv.imshow("depth",depth)
        cv.imshow("binary",bin_im._image_data())
        cv.waitKey(1)
