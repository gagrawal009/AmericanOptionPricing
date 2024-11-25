import numpy as np
from .QDApproximation import HalleySolver
from .AmericanPremium import AmericanPremiumCalculator
from .Chebyshev import compute_coefficient
from .FixedPointScheme import FixedPointIterator


def main(K, S, r, q, sigma, tau_max, l, m, n, p, eta, method):
    """
    Main function to compute the American put option premium.

    Parameters:
        K (float): Strike price.
        S (float): Current stock price.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        sigma (float): Volatility.
        tau_max (float): Maximum time to maturity.
        l (int): Number of integration points for fixed-point iteration.
        m (int): Number of fixed-point iterations.
        n (int): Degree of Chebyshev polynomial (number of nodes - 1).
        p (int): Number of integration points for premium calculation.
        eta (float): Relaxation parameter for iteration.
        method (str): Integration method ('Gauss-Legendre', 'Tanh-Sinh').

    Returns:
        float: The computed American put option premium.
    """
    # Establish Chebyshev nodes and collocation grids
    range_array = np.arange(0,n+1)
    z = - np.cos(range_array * np.pi / n)
    x = (np.sqrt(tau_max)/2) * (1 + z)
    tau = x**2

    # Initialize B_tau for i = 1 to n using QD+ approxiamtion, solved by Halley's method
    B_tau = np.zeros(n+1)
    B_tau[0] = K * np.minimum(1, r / q)
    solver = HalleySolver(K=K, r=r, q=q, sigma=sigma)
    B_tau[1:] = solver.solve_for_B(tau[1:], n_jobs=-1)

    # Fixed Point Scheme iterations
    for j in range(m):
        # Compute Chebyshev coefficients based on B_tau in the last iteration
        f_values = (np.log(B_tau / B_tau[0])) ** 2
        a = compute_coefficient(f_values)

        # Conduct one fixed point scheme iteration
        FP = FixedPointIterator(
            sigma=sigma, r=r, q=q, K=K,
            a=a, tau_max=tau_max, B_tau0=B_tau[0],
            tau_array=tau[1:], B_tau_array=B_tau[1:],
            n_points=l, method=method
        )
        if j == 0:
            # Jacobi-Newton iteration
            B_tau[1:] = FP.iteration(Jacobi_Newton=True, eta=eta, n_jobs=-1)
        else:
            # ordinary Richardson iteration
            B_tau[1:] = FP.iteration(Jacobi_Newton=False, eta=eta, n_jobs=-1)
        eta = eta * 0.5

    # Compute the American put primium
    AP = AmericanPremiumCalculator(
        sigma=sigma, r=r, q=q, S=S, K=K,
        a=a, tau_max=tau_max, B_tau0=B_tau[0],
        tau=tau_max, n_points=p, method='Tanh-Sinh'
    )
    v_tau_max = AP.premium()

    return v_tau_max


if __name__ == '__main__':
    V = main(
        K=100., S=100., r=0.05, q=0.05, sigma=0.25, tau_max=1,
        l=201, m=16, n=64, p=201, eta=0.01, method='Tanh-Sinh'
    )
    print(V)
