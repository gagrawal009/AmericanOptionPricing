import numpy as np
from scipy.stats import norm
from .Integration import NumericalIntegrator
from .Chebyshev import interpolate_B
from joblib import Parallel, delayed


class FixedPointIterator:
    def __init__(self, sigma, r, q, K, a, tau_max, B_tau0, tau_array, B_tau_array, n_points, method):
        """
        Initializes the fixed point iterator class with common parameters.

        Parameters:
            sigma (float): The volatility parameter.
            r (float): The risk-free interest rate.
            q (float): The dividend yield.
            K (float or int): The strike price.
            a (array-like): Coefficients of the Chebyshev series.
            tau_max (float): The maximum horizon of the interval.
            B_tau0 (float): The initial value of B at tau0.
            tau_array (array-like): Array of tau values.
            B_tau_array (array-like): Array of boundary values at tau.
            n_points (int): Number of integration points.
            method (str): Integration method ('Gauss-Legendre', 'Tanh-Sinh').
        """
        self.sigma = sigma
        self.r = r
        self.q = q
        self.K = K
        self.a = a
        self.tau_max = tau_max
        self.B_tau0 = B_tau0
        self.tau_array = np.array(tau_array)
        self.B_tau_array = np.array(B_tau_array)
        self.n_points = n_points
        self.method = method

    def d_plus_minus(self, s, z, sign):
        """
        Vectorized calculation of d+ or d- values.

        Parameters:
            s (array-like): Time to maturity.
            z (array-like): Normally a ratio of two prices.
            sign (int): +1 for d+, -1 for d-.

        Returns:
            ndarray: The calculated d+ or d- values.
        """
        log_term = np.log(z)
        drift = (self.r - self.q + sign * 0.5 * self.sigma**2) * s
        denominator = self.sigma * np.sqrt(s)

        return (log_term + drift) / denominator

    def kappa1(self, tau, B_tau):
        """
        Calculates the kappa1 function for a given tau and B_tau.

        Returns:
            float: The value of kappa1.
        """
        def integrand(y):
            t = tau * (1 + y)**2 / 4
            t_diff = tau - t
            t_diff = np.maximum(1e-10, t_diff)
            exp_term = np.exp(-self.q * t)
            B_tau_minus_t = interpolate_B(tau=t_diff, a=self.a, tau_max=self.tau_max, B_tau0=self.B_tau0)
            B_ratio = B_tau / B_tau_minus_t
            d_plus = self.d_plus_minus(s=t, z=B_ratio, sign=1)
            return exp_term * (1 + y) * norm.cdf(d_plus)

        integrator = NumericalIntegrator(function=integrand, a=-1, b=1)
        I = integrator.integrate(n_points=self.n_points, method=self.method)

        return 0.5 * tau * np.exp(self.q * tau) * I

    def kappa2(self, tau, B_tau):
        """
        Calculates the kappa2 function for a given tau and B_tau.

        Returns:
            float: The value of kappa2.
        """
        def integrand(y):
            t = tau * (1 + y)**2 / 4
            t_diff = tau - t
            t_diff = np.maximum(1e-10, t_diff)
            exp_term = np.exp(-self.q * t)
            B_tau_minus_t = interpolate_B(tau=t_diff, a=self.a, tau_max=self.tau_max, B_tau0=self.B_tau0)
            B_ratio = B_tau / B_tau_minus_t
            d_plus = self.d_plus_minus(s=t, z=B_ratio, sign=1)
            return exp_term / self.sigma * norm.pdf(d_plus)

        integrator = NumericalIntegrator(function=integrand, a=-1, b=1)
        I = integrator.integrate(n_points=self.n_points, method=self.method)

        return np.exp(self.q * tau) * np.sqrt(tau) * I

    def kappa3(self, tau, B_tau):
        """
        Calculates the kappa3 function for a given tau and B_tau.

        Returns:
            float: The value of kappa3.
        """
        def integrand(y):
            t = tau * (1 + y)**2 / 4
            t_diff = tau - t
            t_diff = np.maximum(1e-10, t_diff)
            exp_term = np.exp(-self.r * t)
            B_tau_minus_t = interpolate_B(tau=t_diff, a=self.a, tau_max=self.tau_max, B_tau0=self.B_tau0)
            B_ratio = B_tau / B_tau_minus_t
            d_minus = self.d_plus_minus(s=t, z=B_ratio, sign=-1)
            return exp_term / self.sigma * norm.pdf(d_minus)

        integrator = NumericalIntegrator(function=integrand, a=-1, b=1)
        I = integrator.integrate(n_points=self.n_points, method=self.method)

        return np.exp(self.r * tau) * np.sqrt(tau) * I
    
    def f_tau_single(self, tau, B_tau, Jacobi_Newton=True):
        """
        Calculates the f and f_prime functions given tau.

        Parameters:
            Jacobi_Newton (bool): If True, use Jacobi_Newton scheme; else, use ordinary Richardson scheme.

        Returns:
            float, float: The value of f and f_prime.
        """
        tau = np.maximum(1e-10, tau)
        kappa1 = self.kappa1(tau, B_tau)
        kappa2 = self.kappa2(tau, B_tau)
        kappa3 = self.kappa3(tau, B_tau)

        ratio = B_tau / self.K
        d_minus = self.d_plus_minus(s=tau, z=ratio, sign=-1)
        d_plus = self.d_plus_minus(s=tau, z=ratio, sign=1)

        N_tau = norm.pdf(d_minus) / (self.sigma * np.sqrt(tau)) + self.r * kappa3
        D_tau = norm.cdf(d_plus) + norm.pdf(d_plus) / (self.sigma * np.sqrt(tau)) + self.q * (kappa1 + kappa2)

        coefficient = self.K * np.exp(-(self.r - self.q) * tau)
        f_tau = coefficient * N_tau / D_tau

        f_prime_tau = 0
        if Jacobi_Newton:
            N_prime_tau = -d_minus * norm.pdf(d_minus) / (tau * B_tau * self.sigma**2)
            D_prime_tau = norm.pdf(d_plus) * (self.sigma * np.sqrt(tau) - d_plus) / (tau * B_tau * self.sigma**2)
            f_prime_tau = coefficient * (N_prime_tau / D_tau - (D_prime_tau * N_tau) / (D_tau**2))

        return f_tau, f_prime_tau
    
    def iteration(self, Jacobi_Newton=True, eta=1.0, n_jobs=1):
        """
        Calculates the iterated values of B_tau using the Jacobi-Newton scheme for all tau values.

        Parameters:
            Jacobi_Newton (bool): If True, use Jacobi-Newton scheme; else, use ordinary Richardson scheme.
            eta (float): Hyper-parameter for the iteration scheme.
            n_jobs (int): Number of parallel jobs.

        Returns:
            ndarray: The iterated values of B_tau.
        """
        # Function to process a single tau and B_tau
        def process_single(tau, B_tau):
            f_tau, f_prime_tau = self.f_tau_single(tau, B_tau, Jacobi_Newton=Jacobi_Newton)
            step_size = (B_tau - f_tau) / (f_prime_tau - 1 + 1e-8)  # Added small epsilon to avoid division by zero

            # # TODO: Tuning for hyper-parameter eta.
            # eta = 1.0

            return B_tau + eta * step_size

        # Parallel processing
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single)(tau, B_tau) for tau, B_tau in zip(self.tau_array, self.B_tau_array)
        )

        return np.array(results)
