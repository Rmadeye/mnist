import numpy as np

def plot_from_tensor(tensor: np.ndarray, ax, title: str = ''):
    ax.imshow(tensor, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    return ax


def calc_shape(x: np.ndarray, kernel_size: int = 3, stride: int = 1, padding: int = 0, batch_size: int = 32, channels: int = 1) -> tuple:
    
    #  output size  = (input_size - kernel_size + 2*padding)/stride + 1
    pixel =  (x.shape[2] - kernel_size + 2*padding)//stride + 1
    return (batch_size, channels, pixel, pixel)