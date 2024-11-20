import numpy as np
from scipy.stats import norm
from algorithm1 import numerical_integration

def d_plus_minus(t, B_tau_i, B_tau_minus_t, sigma, r, q, sign):
    B_ratio = B_tau_i / B_tau_minus_t
    log_term = np.log(B_ratio)
    drift = (r - q + sign * sigma**2 / 2) * t
    denominator = sigma * np.sqrt(t)
    return (log_term + drift) / denominator

def kappa1_function(t, tau_i, B_tau_i, B_tau_minus_t, sigma, r, q, l, quadrature_rule):
    def f(y):
        exp_term = np.exp(-q * t)
        d_plus = d_plus_minus(t, B_tau_i, B_tau_minus_t, sigma, r, q, 1)
        return exp_term * (1 + y) * norm.cdf(d_plus)
    
    I = numerical_integration (f, l, quadrature_rule, a=-1, b=1)
    return (tau_i * np.exp(q*tau_i) /2) * I


def kappa2_function(t, tau_i, B_tau_i, B_tau_minus_t, sigma, r, q, l, quadrature_rule):
    def f(y):
        exp_term = np.exp(-q * t)
        d_plus = d_plus_minus(t, B_tau_i, B_tau_minus_t, sigma, r, q, 1)
        return (exp_term/sigma) * norm.cdf(d_plus) 
    
    I = numerical_integration (f, l, quadrature_rule, a=-1, b=1)
    return np.exp(q*tau_i) * np.sqrt(tau_i) * I


def kappa3_function(t, tau_i, B_tau_i, B_tau_minus_t, sigma, r, q, l, quadrature_rule):
    def f(y):
        exp_term = np.exp(-r * t)
        d_plus = d_plus_minus(t, B_tau_i, B_tau_minus_t, sigma, r, q, -1)
        return (exp_term/sigma) * norm.cdf(d_plus) 
    
    I = numerical_integration (f, l, quadrature_rule, a=-1, b=1)
    return np.exp(r*tau_i) * np.sqrt(tau_i) * I




