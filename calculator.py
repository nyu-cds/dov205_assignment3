"""
    Name: Danny Vilela (dov205)
    
    Original runtime: Average(1.37s, 1.38s, 1.38s) =  1.376667s
    Improved runtime: Average(31ms, 31.1ms, 31ms)  = 31.033333ms = 0.0310s
    Relative speedup: (1.376667 / 0.0310) = 44.40

    Comments:
        The majority of the task was using NumPy-native functions
        as opposed to using user-defined functions. Under the hood,
        NumPy uses numerous optimizations in their dot(), add(), and sqrt()
        functions. 
        
        That said, there's no reason to reinvent the wheel. The explicit
        changes were as follow:

        multiple() -> np.multiply()
        add()      -> np.add()
        sqrt()     -> np.sqrt()

    Evaluated from IPython shell with the following:

        %%timeit
        %run calculator.py

    I also modified the default behavior of running `python calculator.py`
    from the terminal in order to properly test the runtime of hypotenuse().
    I figure this is harmless.
"""

# -----------------------------------------------------------------------------
# calculator.py
# ----------------------------------------------------------------------------- 

def hypotenuse(x,y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    xx = multiply(x, x)
    yy = multiply(y, y)
    zz = add(xx, yy)
    return sqrt(zz)


# -----------------------------------------------------------------------------
# calculator_test.py
# ----------------------------------------------------------------------------- 
if __name__ == '__main__':
    
    from numpy import multiply, add, sqrt 
    from numpy.random import random

    M = 10**3
    N = 10**3

    A = random((M,N))
    B = random((M,N))
    C = hypotenuse(A,B)
