import numpy as np
from scipy.stats import norm
from Integration import NumericalIntegrator
from Chebyshev import interpolate_B


class AmericanPremiumCalculator:
    def __init__(self, sigma, r, q, S, K, a, tau_max, B_tau0, tau, n_points, method):
        """
        Initializes the class with common parameters.

        Parameters:
            sigma (float): The volatility parameter.
            r (float): The risk-free interest rate.
            q (float): The dividend yield.
            S (float or int): Current stock price.
            K (float or int): The strike price.
            a (array-like): Coefficients of the Chebyshev series.
            tau_max (float): The maximum horizon of the interval.
            B_tau0 (float): The initial value of B at tau0.
            tau (float): Specific tau value.
            n_points (int): Number of integration points.
            method (str): Integration method ('Gauss-Legendre', 'Tanh-Sinh').
        """
        self.sigma = sigma
        self.r = r
        self.q = q
        self.S = S
        self.K = K
        self.a = a
        self.tau_max = tau_max
        self.B_tau0 = B_tau0
        self.tau = tau
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

    def integral1(self):
        """
        Calculates the first integral function.

        Returns:
            float: The value of integral.
        """
        def integrand(z):
            t = z**2
            t_diff = self.tau - t
            t_diff = np.maximum(1e-10, t_diff)
            exp_term = np.exp(-self.r * t)
            B_tau_minus_t = interpolate_B(tau=t_diff, a=self.a, tau_max=self.tau_max, B_tau0=self.B_tau0)
            B_ratio = self.S / B_tau_minus_t
            d_minus = self.d_plus_minus(s=t, z=B_ratio, sign=-1)
            return z * exp_term * norm.cdf(-d_minus)

        b_right = np.sqrt(self.tau)
        integrator = NumericalIntegrator(function=integrand, a=0, b=b_right)
        I = integrator.integrate(n_points=self.n_points, method=self.method)

        return 2 * self.r * self.K * I
    
    def integral2(self):
        """
        Calculates the second integral function.

        Returns:
            float: The value of integral.
        """
        def integrand(z):
            t = z**2
            t_diff = self.tau - t
            t_diff = np.maximum(1e-10, t_diff)
            exp_term = np.exp(-self.q * t)
            B_tau_minus_t = interpolate_B(tau=t_diff, a=self.a, tau_max=self.tau_max, B_tau0=self.B_tau0)
            B_ratio = self.S / B_tau_minus_t
            d_plus = self.d_plus_minus(s=t, z=B_ratio, sign=1)
            return z * exp_term * norm.cdf(-d_plus)

        b_right = np.sqrt(self.tau)
        integrator = NumericalIntegrator(function=integrand, a=0, b=b_right)
        I = integrator.integrate(n_points=self.n_points, method=self.method)

        return 2 * self.q * self.S * I
    
    def premium(self):
        """
        Calculates the value of American put premium.

        Returns:
            float: The value of the premium.
        """
        integral1_value = self.integral1()
        integral2_value = self.integral2()
        return integral1_value - integral2_value
    