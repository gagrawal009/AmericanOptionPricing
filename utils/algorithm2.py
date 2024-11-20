import numpy as np


def chebyshev_approximation(z, a):
    """
        Performs Chebyshev approximation by Clenshaw algorithm.

        Parameters:
            z (float or ndarray): The point(s) at which to evaluate the Chebyshev series.
            a (array-like): Coefficients of the Chebyshev series, ordered from degree 0 to n.

        Returns:
            float or ndarray: The value(s) of the Chebyshev series at point(s) z.
    """
    n = len(a) - 1
    b = np.zeros(n + 2)
    b[n] = a[n]
    for k in range(n - 1, 0, -1):
        b[k] = a[k] + 2 * z * b[k+1] - b[k+2]
    
    fc = a[0] + b[1] * z - b[2]
    return fc
