import numpy as np
from utils.AmericanPremium import AmericanPrimiumCalculator
from utils.Chebyshev import interpolate_B
from utils.FixedPointScheme import FixedPointIterator
from scipy.stats import norm


def main(K, S, r, q, sigma, tau_max, l, m, n, p, eta, method):
    range_array = np.arange(0,n+1)
    z = - np.cos(range_array * np.pi / n)
    x = (np.sqrt(tau_max)/2) * (1 + z)
    tau = x**2

    B_tau = np.zeros(n+1)
    B_tau[0] = K * np.min(1, r/q)
    
    # TODO:Initialize B_tau for i = 1 to n using eq 7

    for j in range(m):
        # TODO: write 10, 11
        a = np.zeros(n)

        for i in range(1,n+1):
            FP = FixedPointIterator(
                sigma=sigma, r=r, q=q, K=K,
                a=a, tau_max=tau_max, B_tau0=B_tau[0],
                tau=tau[i], B_tau=B_tau[i],
                n_points=l, method=method
            )
            if j == 0:
                B_tau[i] = FP.iteration(Jacobi_Newton=True)
            else:
                B_tau[i] = FP.iteration(Jacobi_Newton=False)

    AP = AmericanPrimiumCalculator(
        sigma=sigma, r=r, q=q, S=S, K=K,
        a=a, tau_max=tau_max, B_tau0=B_tau[0],
        tau=tau_max, n_points=p, method='Tanh-Sinh'
    )
    v_tau_max = AP.premium()

    return v_tau_max
