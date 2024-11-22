import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

tau = 1    # Time to expiry in years
K = 100      # Strike price
r = 0.05     # Risk-free rate
q = 0.05     # Dividend yield
sigma = 0.25  # Volatility

def f(B):
    # Ensure B is positive
    if B <= 0:
        return np.inf

    # Compute d_{+} and d_{-}
    d_plus = (np.log(B / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d_minus = d_plus - sigma * np.sqrt(tau)

    # Compute N(-d_{+}) and N(-d_{-})
    N_minus_d_plus = norm.cdf(-d_plus)
    N_minus_d_minus = norm.cdf(-d_minus)

    # European put option price v(τ, B)
    v = K * np.exp(-r * tau) * N_minus_d_minus - B * np.exp(-q * tau) * N_minus_d_plus

    # Compute phi(d_{+})
    phi_d_plus = norm.pdf(d_plus)

    # Theta of the European put option
    Theta = (r * K * np.exp(-r * tau) * N_minus_d_minus -
             q * B * np.exp(-q * tau) * N_minus_d_plus -
             (sigma * B) / (2 * np.sqrt(tau)) * np.exp(-q * tau) * phi_d_plus)

    # Compute h and omega
    h = 1 - np.exp(-r * tau)
    omega = 2 * (r - q) / sigma**2

    # Compute lambda (λ) and its derivative λ'
    temp = (omega - 1)**2 + 8 * r / (sigma**2 * h)
    sqrt_temp = np.sqrt(temp)
    lam = (- (omega - 1) - sqrt_temp) / 2
    lam_prime = (2 * r) / (sigma**2 * h**2 * sqrt_temp)

    # Compute c0
    numerator = (1 - h) * (2 * r / sigma**2)
    denominator = 2 * lam + omega - 1
    if denominator == 0:
        return np.inf
    term1 = (1 / h)
    denominator2 = r * (K - B - v)
    if denominator2 == 0:
        return np.inf
    term2 = (np.exp(r * tau) * Theta) / denominator2
    term3 = lam_prime / (2 * lam + omega - 1)
    c0 = - numerator / denominator * (term1 - term2 + term3)

    # Compute the left-hand side of the implicit equation
    LHS = - np.exp(-q * tau) * N_minus_d_plus + ((lam + c0) * (K - B - v)) / B

    # Compute f(B)
    f_B = LHS + 1

    return f_B

def f_prime(B):
    # Numerical derivative of f(B) (first derivative)
    h = 1e-6  # Small increment for numerical differentiation
    return (f(B + h) - f(B - h)) / (2 * h)

def f_double_prime(B):
    # Numerical second derivative of f(B) (second derivative)
    h = 1e-6  # Small increment for numerical differentiation
    return (f(B + h) - 2 * f(B) + f(B - h)) / (h**2)

def halley_method(f, f_prime, f_double_prime, B0, tol=1e-8, max_iter=100):
    B = B0
    for i in range(max_iter):
        f_val = f(B)
        f_prime_val = f_prime(B)
        f_double_prime_val = f_double_prime(B)
        
        # Halley's update formula
        denominator = 2 * f_prime_val**2 - f_val * f_double_prime_val
        if denominator == 0:
            raise ValueError("Denominator became zero during Halley's method iteration.")
        
        delta_B = (2 * f_val * f_prime_val) / denominator
        B = B - delta_B
        
        # Check for convergence
        if abs(delta_B) < tol:
            return B
    raise ValueError("Halley's method did not converge within the maximum number of iterations.")

# Initial guess for B(τ)
B0 = 80  # You can adjust this based on the problem context

# Solve for B(τ)
try:
    B_tau = halley_method(f, f_prime, f_double_prime, B0)
    print(f"Solved B(τ) = {B_tau}")
except ValueError as e:
    print(f"Error: {e}")
