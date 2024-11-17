import numpy as np
def numerical_integration (f, l, quadrature_rule, a=-1, b=1):
    S = 0
    if quadrature_rule=='Gauss-Legendre':
        for k in range(1,l+1):
            u = 1
            w = 1
            v = ((b-a)/2) * u + (a+b)/2
            S = S + w * f(v)

    elif quadrature_rule=='Tanh-Sinh':
        print(quadrature_rule)
        s = int(l/2)
        for k in range (-s,s+1):
            u = np.tanh((np.pi/2) * np.sinh(k))
            w = (np.pi/2 * np.cosh(k)) / ((np.cosh((np.pi/2) * np.sinh(k)))**2)
            v = ((b-a)/2) * u + (a+b)/2
            S = S + w * f(v)

    return ((b-a)/2)*S