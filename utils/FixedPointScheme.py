import numpy as np
from scipy.stats import norm
from utils.integration import NumericalIntegrator
from utils.Chebyshev import interpolate_B


class FixedPointIterator:
    def __init__(self, sigma, r, q, K, a, tau_max, B_tau0, tau, B_tau, n_points, method):
        """
        Initializes the KappaFunctions class with common parameters.

        Parameters:
            sigma (float): The volatility parameter.
            r (float): The risk-free interest rate.
            q (float): The dividend yield.
            K (float or int): The strike price.
            a (array-like): Coefficients of the Chebyshev series.
            tau_max (float): The maximum horizon of the interval.
            B_tau0 (float): The initial value of B at tau0.
            tau (float): Specific tau value.
            B_tau (float): Boundary value at tau.
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
        self.tau = tau
        self.B_tau = B_tau
        self.n_points = n_points
        self.method = method

    def d_plus_minus(self, s: float, z: float, sign: int):
        """
        Calculates the d+ or d- value used in option pricing formulas.

        Parameters:
            s (float): Time to maturity.
            z (float): Normally a ratio of two prices.
            sign (int): +1 for d+, -1 for d-.

        Returns:
            float: The calculated d+ or d- value.
        """
        log_term = np.log(z)
        drift = (self.r - self.q + sign * 0.5 * self.sigma**2) * s
        denominator = self.sigma * np.sqrt(s)

        return (log_term + drift) / denominator

    def kappa1(self):
        """
        Calculates the kappa1 function.

        Returns:
            float: The value of kappa1.
        """
        def integrand(y):
            t = self.tau * (1 + y)**2 / 4
            t_diff = self.tau - t
            exp_term = np.exp(-self.q * t)
            B_tau_minus_t = interpolate_B(tau=t_diff, a=self.a, tau_max=self.tau_max, B_tau0=self.B_tau0)
            B_ratio = self.B_tau / B_tau_minus_t
            d_plus = self.d_plus_minus(s=t, z=B_ratio, sign=1)
            return exp_term * (1 + y) * norm.cdf(d_plus)

        integrator = NumericalIntegrator(function=integrand, a=-1, b=1)
        I = integrator.integrate(n_points=self.n_points, method=self.method)

        return 0.5 * self.tau * np.exp(self.q * self.tau) * I

    def kappa2(self):
        """
        Calculates the kappa2 function.

        Returns:
            float: The value of kappa2.
        """
        def integrand(y):
            t = self.tau * (1 + y)**2 / 4
            t_diff = self.tau - t
            exp_term = np.exp(-self.q * t)
            B_tau_minus_t = interpolate_B(tau=t_diff, a=self.a, tau_max=self.tau_max, B_tau0=self.B_tau0)
            B_ratio = self.B_tau / B_tau_minus_t
            d_plus = self.d_plus_minus(s=t, z=B_ratio, sign=1)
            return exp_term / self.sigma * norm.pdf(d_plus)

        integrator = NumericalIntegrator(function=integrand, a=-1, b=1)
        I = integrator.integrate(n_points=self.n_points, method=self.method)

        return np.exp(self.q * self.tau) * np.sqrt(self.tau) * I

    def kappa3(self):
        """
        Calculates the kappa3 function.

        Returns:
            float: The value of kappa3.
        """
        def integrand(y):
            t = self.tau * (1 + y)**2 / 4
            t_diff = self.tau - t
            exp_term = np.exp(-self.r * t)
            B_tau_minus_t = interpolate_B(tau=t_diff, a=self.a, tau_max=self.tau_max, B_tau0=self.B_tau0)
            B_ratio = self.B_tau / B_tau_minus_t
            d_minus = self.d_plus_minus(s=t, z=B_ratio, sign=-1)
            return exp_term / self.sigma * norm.pdf(d_minus)

        integrator = NumericalIntegrator(function=integrand, a=-1, b=1)
        I = integrator.integrate(n_points=self.n_points, method=self.method)

        return np.exp(self.r * self.tau) * np.sqrt(self.tau) * I
    
    def f_tau(self, Jacobi_Newton: bool = True):
        """
        Calculates the f and f_prime functions given tau.

        Parameters:
            Jacobi_Newton (bool): If True, use Jacobi_Newton scheme; else, use ordinary Richardson scheme.

        Returns:
            float, float: The value of f and f_prime.
        """
        kappa1 = self.kappa1()
        kappa2 = self.kappa2()
        kappa3 = self.kappa3()

        ratio = self.B_tau / self.K
        d_minus = self.d_plus_minus(s=self.tau, z=ratio, sign=-1)
        d_plus = self.d_plus_minus(s=self.tau, z=ratio, sign=1)

        N_tau = norm.pdf(d_minus) / (self.sigma * np.sqrt(self.tau)) + self.r * kappa3
        D_tau = norm.cdf(d_plus) + norm.pdf(d_plus) / (self.sigma * np.sqrt(self.tau)) + self.q * (kappa1 + kappa2)

        coefficient = self.K * np.exp(-(self.r - self.q) * self.tau)
        f_tau = coefficient * self.N_tau() / self.D_tau()

        f_prime_tau = 0
        if Jacobi_Newton:
            N_prime_tau = -d_minus * norm.pdf(d_minus) / (self.tau * self.B_tau * self.sigma**2)
            D_prime_tau = -d_minus * norm.pdf(d_plus) / (self.tau * self.B_tau * self.sigma**2)
            f_prime_tau = coefficient * (N_prime_tau / D_tau - (D_prime_tau * N_tau) / (D_tau**2))

        return f_tau, f_prime_tau
    
    def iteration(self, Jacobi_Newton: bool = True, eta: float = 1.0):
        """
        Calculates the iterated value of B_tau using the Jacobi-Newton scheme.

        Parameters:
            Jacobi_Newton (bool): If True, use Jacobi_Newton scheme; else, use ordinary Richardson scheme.
            eta (float): hyper-parameter for the iteration scheme.

        Returns:
            float: The iterated value of B_tau.
        """

        f_tau, f_prime_tau = self.f_tau(Jacobi_Newton=Jacobi_Newton)
        step_size = (self.B_tau - f_tau) / (f_prime_tau - 1)
        eta = 1.0

        # TODO: Tuning for hyper-parameter eta.
        
        return self.B_tau + eta * step_size
