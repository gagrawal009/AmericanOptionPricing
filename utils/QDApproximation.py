import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from joblib import Parallel, delayed

class HalleySolver:
    def __init__(self, K, r, q, sigma):
        """
        Initializes the HalleySolver with option parameters.

        Parameters:
            K (float): Strike price.
            r (float): Risk-free interest rate.
            q (float): Dividend yield.
            sigma (float): Volatility.
        """
        self.K = K
        self.r = r
        self.q = q
        self.sigma = sigma

    def f(self, B, tau):
        """
        Computes the function f(B) used in Halley's method.

        Parameters:
            B (float): The variable for which we are solving.
            tau (float): Time to expiry in years.

        Returns:
            float: The value of the function f(B).
        """
        # Ensure B is positive
        if B <= 0:
            raise ValueError("B must be positive.")

        K = self.K
        r = self.r
        q = self.q
        sigma = self.sigma

        # Compute d_{+} and d_{-}
        sqrt_tau = np.sqrt(tau)
        sigma_sqrt_tau = sigma * sqrt_tau
        log_term = np.log(B / K)
        d_plus = (log_term + (r - q + 0.5 * sigma**2) * tau) / sigma_sqrt_tau
        d_minus = d_plus - sigma_sqrt_tau

        # Compute N(-d_{+}) and N(-d_{-})
        N_minus_d_plus = norm.cdf(-d_plus)
        N_minus_d_minus = norm.cdf(-d_minus)

        # European put option price v(τ, B)
        exp_neg_r_tau = np.exp(-r * tau)
        exp_neg_q_tau = np.exp(-q * tau)
        v = K * exp_neg_r_tau * N_minus_d_minus - B * exp_neg_q_tau * N_minus_d_plus

        # Compute phi(d_{+})
        phi_d_plus = norm.pdf(d_plus)

        # Theta of the European put option
        Theta = (
            r * K * exp_neg_r_tau * N_minus_d_minus
            - q * B * exp_neg_q_tau * N_minus_d_plus
            - (sigma * B) / (2 * sqrt_tau) * exp_neg_q_tau * phi_d_plus
        )

        # Compute h and omega
        h = 1 - exp_neg_r_tau
        if h == 0:
            raise ValueError("h cannot be zero.")
        omega = 2 * (r - q) / sigma**2

        # Compute lambda (λ) and its derivative λ'
        temp = (omega - 1) ** 2 + (8 * r) / (sigma**2 * h)
        if temp < 0:
            raise ValueError("Negative discriminant encountered.")
        sqrt_temp = np.sqrt(temp)
        lam = (- (omega - 1) - sqrt_temp) / 2
        if lam == 0:
            raise ValueError("Lambda (λ) cannot be zero.")
        lam_prime = (2 * r) / (sigma**2 * h ** 2 * sqrt_temp)

        # Compute c0
        numerator = (1 - h) * (2 * r / sigma**2)
        denominator = 2 * lam + omega - 1
        if denominator == 0:
            raise ValueError("Denominator in c0 computation is zero.")
        term1 = 1 / h
        denominator2 = r * (K - B - v)
        if denominator2 == 0:
            raise ValueError("Denominator2 in c0 computation is zero.")
        term2 = (np.exp(r * tau) * Theta) / denominator2
        term3 = lam_prime / denominator
        c0 = -numerator / denominator * (term1 - term2 + term3)

        # Compute the left-hand side of the implicit equation
        LHS = -exp_neg_q_tau * N_minus_d_plus + ((lam + c0) * (K - B - v)) / B

        # Compute f(B)
        f_B = LHS + 1

        return f_B

    def f_prime(self, B, tau):
        """
        Computes the first derivative of f(B) numerically.

        Parameters:
            B (float): The variable for which we are computing the derivative.
            tau (float): Time to expiry in years.

        Returns:
            float: The first derivative of f(B) with respect to B.
        """
        h = 1e-5  # Adjusted increment for better numerical stability
        return (self.f(B + h, tau) - self.f(B - h, tau)) / (2 * h)

    def f_double_prime(self, B, tau):
        """
        Computes the second derivative of f(B) numerically.

        Parameters:
            B (float): The variable for which we are computing the second derivative.
            tau (float): Time to expiry in years.

        Returns:
            float: The second derivative of f(B) with respect to B.
        """
        h = 1e-4  # Larger increment for second derivative
        return (self.f(B + h, tau) - 2 * self.f(B, tau) + self.f(B - h, tau)) / (h ** 2)

    def compute_default_B0(self, tau_array):
        """
        Computes default initial guesses for B0 based on the option parameters.

        Parameters:
            tau_array (float or ndarray): Time to expiry in years.

        Returns:
            ndarray: The default initial guesses for B0.
        """
        tau_array = np.array(tau_array, dtype=np.float64)
        sigma_sq = self.sigma ** 2
        r_minus_q = self.r - self.q

        sqrt_term = np.sqrt(
            ((r_minus_q) / sigma_sq - 0.5) ** 2 + 2 * self.r / sigma_sq
        )
        beta_2 = (0.5 - (r_minus_q) / sigma_sq) + sqrt_term

        # Perpetual early exercise boundary
        B_infinite = self.K * (beta_2 / (beta_2 - 1))

        # Adjust initial guess based on time to expiry
        weight = np.exp(-tau_array)  # Weight decreases with longer tau
        B0 = weight * self.K + (1 - weight) * B_infinite

        # Ensure B0 is between 0.01 and K
        B0 = np.clip(B0, 0.01, self.K)

        return B0

    def solve_for_B(self, tau_array, B0_array=None, tol=1e-8, max_iter=100, n_jobs=1):
        """
        Solves for B(τ) over an array of τ values.

        Parameters:
            tau_array (array-like): Array of time to expiry values.
            B0_array (array-like, optional): Initial guesses for B. If not provided, default values are computed.
            tol (float, optional): Tolerance for convergence.
            max_iter (int, optional): Maximum number of iterations.
            n_jobs (int, optional): Number of jobs for parallel computation.

        Returns:
            ndarray: Array of solved values of B(τ).

        Raises:
            RuntimeError: If the method fails to converge.
        """
        tau_array = np.array(tau_array, dtype=np.float64)
        if B0_array is None:
            B0_array = self.compute_default_B0(tau_array)

        # Function to solve for a single τ
        def solve_single_B(tau, B0):
            try:
                B_solution = newton(
                    func=lambda B: self.f(B, tau),
                    x0=B0,
                    fprime=lambda B: self.f_prime(B, tau),
                    fprime2=lambda B: self.f_double_prime(B, tau),
                    tol=tol,
                    maxiter=max_iter,
                )
                return B_solution
            except (RuntimeError, ValueError) as e:
                print(f"Root-finding failed for tau={tau}: {e}")
                return np.nan  # Return NaN if root-finding fails

        # Parallel computation over τ values
        results = Parallel(n_jobs=n_jobs)(
            delayed(solve_single_B)(tau, B0) for tau, B0 in zip(tau_array, B0_array)
        )

        return np.array(results)

# Usage example
if __name__ == "__main__":
    # Option parameters
    K = 100       # Strike price
    r = 0.05      # Risk-free rate
    q = 0.05      # Dividend yield
    sigma = 0.25  # Volatility

    # Instantiate the solver
    solver = HalleySolver(K, r, q, sigma)

    # Array of τ values
    tau_array = np.linspace(0.01, 1.0, 100)  # 100 points from 0.01 to 1.0 years

    # Solve for B(τ) over tau_array
    B_tau_array = solver.solve_for_B(tau_array, n_jobs=-1)  # Use all available cores
    print(f"Solved initial B(τ) = {B_tau_array[-1]}")

    # Plot the results
    import matplotlib.pyplot as plt

    plt.plot(tau_array, B_tau_array)
    plt.xlabel('Time to Expiry τ')
    plt.ylabel('Early Exercise Boundary B(τ)')
    plt.title('Early Exercise Boundary vs. Time to Expiry')
    plt.grid(True)
    plt.show()
