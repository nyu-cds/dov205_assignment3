from numpy import dot, add, sqrt

"""
    Name: Danny Vilela (dov205)
    
    Original runtime: Average(1.37s, 1.38s, 1.38s) = 1.376667s
    Improved runtime: Average(55.9ms, 55ms, 55ms) = 55.300000ms = 0.0553s
    Relative speedup: (1.376667 / 0.0553) = 24.8945

    Comments:
        The majority of the task was using NumPy-native functions
        as opposed to using user-defined functions. Under the hood,
        NumPy uses numerous optimizations in their dot(), add(), and sqrt()
        functions. 
        
        That said, there's no reason to reinvent the wheel. The explicit
        changes were as follow:

        multiple() -> np.dot()
        add()      -> np.add()
        sqrt()     -> np.sqrt()
"""

# -----------------------------------------------------------------------------
# calculator.py
# ----------------------------------------------------------------------------- 
from numpy import dot, add, sqrt


def hypotenuse(x,y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    xx = dot(x, x)
    yy = dot(y, y)
    zz = add(xx, yy)
    return sqrt(zz)


# -----------------------------------------------------------------------------
# calculator_test.py
# ----------------------------------------------------------------------------- 
if __name__ == '__main__':
    
    from numpy.random import random

    M = 10**3
    N = 10**3

    A = random((M,N))
    B = random((M,N))
    C = hypotenuse(A,B)
