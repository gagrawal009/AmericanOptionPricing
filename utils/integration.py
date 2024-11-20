import numpy as np


class NumericalIntegrator:
    """
    A class for performing numerical integration using various quadrature rules.
    """

    def __init__(self, function, a: float = -1.0, b: float = 1.0):
        """
        Initializes the NumericalIntegrator with the function to integrate and the integration bounds.

        Parameters:
            function (callable): The function to integrate.
            a (float, optional): Lower bound of integration. Default is -1.0.
            b (float, optional): Upper bound of integration. Default is 1.0.
        """
        self.function = function
        self.a = a
        self.b = b

    def integrate(self, n_points: int, method: str = 'Gauss-Legendre') -> float:
        """
        Performs numerical integration using the specified quadrature rule.

        Parameters:
            n_points (int): Number of integration points.
            method (str, optional): The quadrature rule to use ('Gauss-Legendre' or 'Tanh-Sinh').
                                    Default is 'Gauss-Legendre'.

        Returns:
            float: The numerical approximation of the integral.
        """
        if method == 'Gauss-Legendre':
            return self._gauss_legendre(n_points)
        elif method == 'Tanh-Sinh':
            return self._tanh_sinh(n_points)
        else:
            raise ValueError("Unsupported method. Choose 'Gauss-Legendre' or 'Tanh-Sinh'.")

    def _gauss_legendre(self, n_points: int) -> float:
        """
        Private method to perform Gauss-Legendre quadrature.

        Parameters:
            n_points (int): Number of integration points.

        Returns:
            float: The numerical approximation of the integral.
        """
        nodes, weights = np.polynomial.legendre.leggauss(n_points)
        # Map the nodes from [-1, 1] to [a, b]
        transformed_nodes = 0.5 * (self.b - self.a) * nodes + 0.5 * (self.a + self.b)
        result = np.dot(weights, self.function(transformed_nodes))
        
        return 0.5 * (self.b - self.a) * result

    def _tanh_sinh(self, n_points: int, h: float = 1.0) -> float:
        """
        Private method to perform Tanh-Sinh quadrature.

        Parameters:
            n_points (int): Number of integration points.
            h (float, optional): Step size can be adjusted as needed. Default is 1.0.

        Returns:
            float: The numerical approximation of the integral.
        """
        s = n_points // 2
        k_values = np.arange(-s, s + 1)
        t = k_values * h
        sinh_t = np.sinh(t)
        cosh_t = np.cosh(t)
        pi_over_2 = np.pi / 2

        u = np.tanh(pi_over_2 * sinh_t)
        w = (pi_over_2 * h * cosh_t) / (np.cosh(pi_over_2 * sinh_t) ** 2)

        # Map the nodes from [-1, 1] to [a, b]
        transformed_nodes = 0.5 * (self.b - self.a) * u + 0.5 * (self.a + self.b)
        result = np.dot(w, self.function(transformed_nodes))

        return 0.5 * (self.b - self.a) * result
