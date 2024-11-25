import numpy as np


def compute_coefficient(f_values):
    """
    Computes the coefficients a_k for Chebyshev interpolation.

    Parameters:
        f_values (ndarray): Array of function values at Chebyshev nodes z_i.

    Returns:
        ndarray: Array of coefficients a_k.
    """
    n = len(f_values) - 1   # Since z_array has n+1 points from z_0 to z_n, so does f(z)
    weights = np.ones(n + 1)
    weights[0] = 0.5   # Halve the first term
    weights[-1] = 0.5  # Halve the last term

    # Initialize the array for coefficients a_k
    a_k = np.zeros(n + 1)

    # Compute a_0
    sum_k0 = np.sum(weights * f_values)
    a_k[0] = (1 / n) * sum_k0

    # Compute a_k for k = 1 to n - 1
    i_array = np.arange(n + 1)  # i from 0 to n
    k_array = np.arange(1, n)   # k from 1 to n - 1
    k_i_matrix = np.outer(k_array, i_array)      # Outer product to get k * i
    cos_matrix = np.cos(k_i_matrix * np.pi / n)  # Compute cos(k * i * pi / n)

    # Multiply weights and f_values
    weighted_f_values = weights * f_values

    # Compute the sums for each k using matrix multiplication
    sums_k = cos_matrix @ weighted_f_values  # Shape (n - 1,)

    # Compute a_k for k = 1 to n - 1
    a_k[1:n] = (2 / n) * sums_k

    # Compute a_n
    signs = (-1) ** i_array  # Compute (-1)^i for i from 0 to n
    sum_kn = np.sum(weights * f_values * signs)
    a_k[n] = (1 / n) * sum_kn

    return a_k



def chebyshev_approximation(z, a):
    """
    Performs Chebyshev approximation by Clenshaw algorithm.

    Parameters:
        z (float or ndarray): The point(s) at which to evaluate the Chebyshev series.
        a (array-like): Coefficients of the Chebyshev series, ordered from degree 0 to n.

    Returns:
        float or ndarray: The value(s) of the Chebyshev series at point(s) z.
    """
    z = np.atleast_1d(z)  # Ensure z is at least 1D array
    n = len(a) - 1
    b = np.zeros((n + 2, len(z)))  # Adjust b to be 2D to handle arrays
    b[n, :] = a[n]
    for k in range(n - 1, 0, -1):
        b[k, :] = a[k] + 2 * z * b[k + 1, :] - b[k + 2, :]
    fc = a[0] + b[1, :] * z - b[2, :]

    # If input z was scalar, return scalar output
    if fc.size == 1:
        return fc[0]
    else:
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
    tau = np.atleast_1d(tau)  # Ensure tau is at least 1D array
    z = 2 * np.sqrt(tau) / np.sqrt(tau_max) - 1
    qc = chebyshev_approximation(z=z, a=a)

    # # Ensure qc is non-negative
    # qc = np.maximum(qc, 0)
    B_tau = B_tau0 * np.exp(np.sqrt(qc))

    # If input tau was scalar, return scalar output
    if B_tau.size == 1:
        return B_tau[0]
    else:
        return B_tau
