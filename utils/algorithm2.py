import numpy as np

def chebyshev_approximation(z, a):
    l = len(a)
    n = l-1
    b = np.zeros(l+1)
    b[n+1] = 0
    b[n] = a[n]
    for k in range(n-1, 0, -1):
        b[k] = a[k] + 2*z*b[k+1] - b[k+2]
    
    fc = a[0] + b[1]*z - b[2]
    return fc


