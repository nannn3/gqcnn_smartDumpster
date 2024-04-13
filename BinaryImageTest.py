import scipy.ndimage.morphology as snm
import numpy as np
from Astra import Astra
import cv2 as cv
from autolab_core import ColorImage, DepthImage, BinaryImage

def filter_binary_im(binary_im,w):
        # Filter function taken from autolab_core detector
        w = w
        y, x = np.ogrid[-w / 2 + 1 : w / 2 + 1, -w / 2 + 1 : w / 2 + 1]
        mask = x * x + y * y <= w / 2 * w / 2
        filter_struct = np.zeros([w, w]).astype(np.uint8)
        filter_struct[mask] = 1
        binary_im_filtered = binary_im.apply(
            snm.grey_closing, structure=filter_struct
        )
        # ===
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
        
        binary_im_filtered = filter_binary_im(final_obj_mask, 5) #TODO replace 5 with better value
    
        
        cv.imshow("color",color)
        # cv.imshow("obj_mask_filtered",binary_im_filtered._image_data())
        cv.imshow("obj_mask_unfiltered",final_obj_mask._image_data())
        #cv.imshow("depth_filtered",filtered_depth_im._image_data())
        #cv.imshow("binary",bin_im._image_data())
        cv.waitKey(1)
