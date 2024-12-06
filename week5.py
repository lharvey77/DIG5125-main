import numpy as np
from scipy.signal import convolve2d

'''def simple_1d_convolution(input_array, conv_mask):
    input_array = np.array([1, 2, 3, 4, 5])
    conv_mask = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) 
    return np.convolve(input_array, conv_mask, mode='full')'''

def simple_2d_convolution(input_array, conv_mask):
    input_array = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
    conv_mask = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1]])
    
    return convolve2d(input_array, conv_mask, mode='full')
output = simple_2d_convolution(input_array, conv_mask, mode='same', boundary='fill')
print(output)