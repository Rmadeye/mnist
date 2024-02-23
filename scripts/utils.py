import numpy as np

def calc_shape(x: np.ndarray, kernel_size: int = 3, stride: int = 1, padding: int = 0, dilation: int  = 1, channels: int = 1):
    return int((x + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) 
