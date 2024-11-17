from algorithm2 import chebyshev_approximation
import numpy as np

def interpolate_transformed_function(tau, a, tau_max, B_tau0):
    z = (2 * (np.sqrt(tau)/np.sqrt(tau_max))) - 1
    qc = chebyshev_approximation(z, a)
    B_tau = B_tau0 * np.exp(np.sqrt(qc))
    return B_tau
