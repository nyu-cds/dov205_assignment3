"""
    Name: Danny Vilela (dov205)
    
    Original runtime: Average(1.37s, 1.38s, 1.38s)    =  1.376667s
    Improved runtime: Average(25.3ms, 25.4ms, 25.3ms) = 25.333333ms = 0.0253333s
    Relative speedup: (1.376667 / 0.0253333) = 54.342

    Comments:
        The majority of the task was using user-defined functions that 
        are (optimally) implemented as NumPy-native functions. In fact,
        during my first iteration of calculator.py I replaced the following:
        
            multiple() -> np.multiply()
            add()      -> np.add()
            sqrt()     -> np.sqrt()

        This had a huge speedup of approximately 44x relative to the base
        implementation. After some investigation, however, it turns out the 
        NumPy API actually has a function for computing the hypotenuse! After
        implementing that solution, I achieved a relative speedup of 54x.

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

# Empty! Who needs it.

# -----------------------------------------------------------------------------
# calculator_test.py
# ----------------------------------------------------------------------------- 
if __name__ == '__main__':
    
    from numpy import hypot
    from numpy.random import random

    M = 10**3
    N = 10**3

    A = random((M,N))
    B = random((M,N))
    C = hypot(A,B)

