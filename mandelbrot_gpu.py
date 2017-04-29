# 
# A CUDA version to calculate the Mandelbrot set
#

from numba import cuda
import numpy as np
from pylab import imshow, show

@cuda.jit(device=True)
def mandel(x, y, max_iters):
    '''
    Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the 
    Mandelbrot set given a fixed number of iterations.
    '''
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@cuda.jit
def compute_mandel(min_x, max_x, min_y, max_y, image, iters):
    """Assign a value to each pixel in the :image array corresponding to its :mandel() value.

    Our CUDA-supported function works as follows:

        1. Obtain the starting x and y coordinates using :cuda.grid()
        2. Calculate the ending x and y coordinates by obtaining the size of 
           the block using gridDim and blockDim.
        3. Calculate the mandel value for each element in our designated partition.
    
    From there, we can utilize the Professor's prior, non-CUDA-supported code 
    to compute the mandel value for each value in the image array. Note: this
    kernel function iterates over a smaller block of the elements within :image
    according on the :cuda.gridDim() and :cuda.blockDim() configurations.
    """

    # Inherited from Professor Watson's source code.
    height = image.shape[0]
    width  = image.shape[1]
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
 
    # Obtain the starting (x, y) coordinates using `cuda.grid()`
    start_x, start_y = cuda.grid(2)

    # Calculate the ending (x, y) coordinates by obtaining the grid and block dimensions.
    end_x = cuda.gridDim.x * cuda.blockDim.x
    end_y = cuda.gridDim.y * cuda.blockDim.y

    # Compute the mandel value for each element in the block, like before.
    for x in range(start_x, width, end_x):
        real = min_x + x * pixel_size_x
        for y in range(start_y, height, end_y):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel(real, imag, iters)


if __name__ == '__main__':
    image = np.zeros((1024, 1536), dtype = np.uint8)
    blockdim = (32, 8)
    griddim = (32, 16)
    
    image_global_mem = cuda.to_device(image)
    compute_mandel[griddim, blockdim](-2.0, 1.0, -1.0, 1.0, image_global_mem, 20) 
    image_global_mem.copy_to_host()
    imshow(image)
    show()
