import numpy as np
from autolab_core import DepthImage,ColorImage
import Astra

def normalize_array(arr):
    """
    Normalize an array to the range [0,1]
    Paramaters:
        numpy.ndarray to be normalized
    Returns:
        numpy.ndarry: The normalized array
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr -min_val)/(max_val - min_val)
    return normalized_arr

if __name__ == '__main__':
    dev = Astra.Astra()
    dev.start()
    color_frame, depth_frame = dev.frames()
    normalized_depth_frame = normalize_array(depth_frame)
    color_im = ColorImage(color_frame)
    depth_im = DepthImage(normalized_depth_frame)
    print(depth_im)
