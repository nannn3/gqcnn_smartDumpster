from Astra import Astra
import cv2 as cv
from autolab_core import ColorImage, DepthImage, BinaryImage

if __name__ == "__main__":
    camera = Astra()
    camera.start()
    while (1):
        color,depth = camera.frames()
        color_im = ColorImage(color)
        depth_im = DepthImage(depth)
        filtered_depth_im = depth_im.threshold(.5,1)

        bin_im = color_im.foreground_mask(50,ignore_black =False)
        depth_mask = filtered_depth_im.invalid_pixel_mask()
        right_depth_mask = depth_mask.inverse()
        final_obj_mask = bin_im.mask_binary(right_depth_mask)
        cv.imshow("color",color)
        cv.imshow("obj_mask",final_obj_mask._image_data())
        #cv.imshow("depth_filtered",filtered_depth_im._image_data())
        #cv.imshow("binary",bin_im._image_data())
        cv.waitKey(1)
