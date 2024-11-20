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

def interpolate_B(tau, a, tau_max, B_tau0):
    """
    Interpolates B for given value of tau in the interval, through chebyshev approximation of transformed function H.

    Parameters:
        tau (float or ndarray): The value(s) of tau at which to interpolate B.
        a (array-like): Coefficients of the Chebyshev series.
        tau_max (float): The maximum horizon of the interval.
        B_tau0 (float): The initial value of B at tau0.

    Returns:
        float or ndarray: The interpolated value(s) of B at the specified tau.
    """
    z = 2 * np.sqrt(tau) / np.sqrt(tau_max) - 1
    qc = chebyshev_approximation(z=z, a=a)
    B_tau = B_tau0 * np.exp(np.sqrt(qc))

    return B_tau
