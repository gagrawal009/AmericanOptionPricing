import numpy as np
from algorithm1 import numerical_integration
from algorithm3 import interpolate_transformed_function
from kappa import kappa1_function, kappa2_function, kappa3_function
from scipy.stats import norm

def d_plus_minus(B_tau_i, tau_i, K, sigma, r, q, sign):
    B_ratio = B_tau_i / K
    log_term = np.log(B_ratio)
    drift = (r - q + sign * sigma**2 / 2) * tau_i
    denominator = sigma * np.sqrt(tau_i)
    return (log_term + drift) / denominator

def N_tau_B(B_tau_i, tau_i, K, sigma, r, q, kappa3):
    d_minus = d_plus_minus(B_tau_i, tau_i, K, sigma, r, q, -1)
    phi = norm.cdf(d_minus)
    return (phi / (sigma*np.sqrt(tau_i))) + r * kappa3

def D_tau_B(B_tau_i, tau_i, K, sigma, r, q, kappa1, kappa2):
    d_plus = d_plus_minus(B_tau_i, tau_i, K, sigma, r, q, 1)
    phi = norm.cdf(d_plus)
    return phi + (phi / (sigma*np.sqrt(tau_i))) + q * (kappa1 + kappa2)

def f_tau_B (tau_i, r, q, K, N, D):
    exp_term = np.exp(-(r-q) * tau_i)
    return K * exp_term * (N/D)

def N_dash_tau_B(B_tau_i, tau_i, K, sigma, r, q):
    d_minus = d_plus_minus(B_tau_i, tau_i, K, sigma, r, q, -1)
    phi = norm.cdf(d_minus)
    return -d_minus * (phi / (B_tau_i * sigma**2 * tau_i))

def D_dash_tau_B(B_tau_i, tau_i, K, sigma, r, q):
    d_minus = d_plus_minus(B_tau_i, tau_i, K, sigma, r, q, -1)
    d_plus = d_plus_minus(B_tau_i, tau_i, K, sigma, r, q, 1)
    phi = norm.cdf(d_plus)
    return -d_minus * (phi / (B_tau_i * sigma**2 * tau_i))

def f_dash_tau_B (tau_i, r, q, K, N_dash, D_dash, N, D):
    exp_term = np.exp(-(r-q) * tau_i)
    return K * exp_term * ((N_dash/D) - ((D_dash*N) / D**2))

def american_option_pricing(K, S, r, q, sigma, tau_max, l, m, n, p, eta, quadrature_rule):
    range_array = np.arange(0,n+1)
    z = - np.cos(range_array * np.pi/n)
    x = (np.sqrt(tau_max)/2) * (1 +z)
    tau = x**2

    y, w = np.polynomial.legendre.leggauss(l)

    B_tau = np.zeros(n+1)
    B_tau[0] = K * np.min(1, r/q)

    #get B_tau for i = 1 to n using eq 7

    for j in range(1,m+1):
        # write 10,11

        for i in range(1,n+1):
            t = tau[i]*np.square(1+y)/4    
            tau_minus_t = tau[i] - t
            B_tau_minus_t = interpolate_transformed_function(tau_minus_t, a, tau_max, B_tau[0])

            kappa1 = kappa1_function(t, tau[i], B_tau[i], B_tau_minus_t, sigma, r, q, l, quadrature_rule)
            kappa2 = kappa2_function(t, tau[i], B_tau[i], B_tau_minus_t, sigma, r, q, l, quadrature_rule)
            kappa3 = kappa3_function(t, tau[i], B_tau[i], B_tau_minus_t, sigma, r, q, l, quadrature_rule)

            N = N_tau_B(B_tau[i], tau[i], K, sigma, r, q, kappa3)
            D = D_tau_B(B_tau[i], tau[i], K, sigma, r, q, kappa1, kappa2)
            f_taui = f_tau_B (tau[i], r, q, K, N, D)
            f_dash_taui = 0

            if (j==1):
                N_dash = N_dash_tau_B(B_tau[i], tau[i], K, sigma, r, q)
                D_dash = D_dash_tau_B(B_tau[i], tau[i], K, sigma, r, q)
                f_dash_taui = f_dash_tau_B (tau[i], r, q, K, N_dash, D_dash, N, D)
            else:
                f_dash_taui = 0
            
            B_tau[i] = B_tau[i] + eta *((B_tau[i] - f_taui)/(f_dash_taui -1))

    z, w = np.polynomial.legendre.leggauss(p)
    tau_p = np.zeros(p)
    B_tau_p = np.zeros(p)
    for k in range(p):
        tau_p[k] = tau_max - z[k]**2
        if tau_p[k] in tau:
            idx = tau.index(tau_p[k])
            B_tau_p[k] = B_tau[idx]
        else:
            #write line 30
            print(1)


    v_tau_max_S = 0     #write line 33
    return v_tau_max_S
