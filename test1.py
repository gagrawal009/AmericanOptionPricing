from utils.CompleteAlgorithm import main
import time

# Fix option parameters, test algorithm parameters.
def test_case(K, S, r, q, sigma, T, eta, parameter_sets):
    """
    Test the performance and accuracy of the `main` function with different parameter sets.

    This function computes the benchmark American option premium using a high-accuracy method
    and compares it against results obtained using varying parameter sets. It measures and
    outputs the relative error and computation time for each parameter set.

    Parameters:
        K (float): Strike price.
        S (float): Current stock price.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        sigma (float): Volatility.
        T (float): Time to maturity.
        eta (float): Relaxation parameter for iteration.
        parameter_sets (list of tuples): A list of parameter sets to test.
            Each tuple should contain the following values:
                - l (int): Number of integration points for fixed-point iteration.
                - m (int): Number of fixed-point iterations.
                - n (int): Degree of Chebyshev polynomial (number of nodes - 1).
                - p (int): Number of integration points for premium calculation.
                - method (str): Integration method ('Gauss-Legendre', 'Tanh-Sinh').
    Outputs:
        Prints:
            - The benchmark American premium.
            - The computed American premium for each parameter set.
            - The relative error compared to the benchmark.
            - The CPU time taken for each computation.

    Returns:
        None: Results are printed to the console.
    """
    benchmark = main(
        K=K, S=S, r=r, q=q, sigma=sigma, tau_max=T,
        l=201, m=16, n=64, p=201, eta=eta, method='Tanh-Sinh'
    )
    print("The benchmark American Premium is: {:.12f}.".format(benchmark))

    for params in parameter_sets:
        l, m, n, p, method = params

        start_time = time.perf_counter()
        V = main(
            K=K, S=S, r=r, q=q, sigma=sigma, tau_max=T,
            l=l, m=m, n=n, p=p, eta=eta, method=method
        )
        end_time = time.perf_counter()
        print(f"(l,m,n)=({l},{m},{n}), p={p}, method={method}")
        print("American Premium: {:.12f}".format(V))
        error = (V - benchmark) / benchmark
        print(f"Relative Error: {error:.2E}")
        elapsed_time = end_time - start_time
        print(f"CPU Seconds: {elapsed_time:.2E}")


if __name__ == '__main__':
    K = 100.
    S = 100.
    r = 0.05
    q = 0.05
    sigma = 0.25
    T = 1
    eta = 0.01
    parameter_sets = [
        (5, 1, 4, 15, 'Gauss-Legendre'),
        (7, 2, 5, 20, 'Gauss-Legendre'),
        (11, 2, 5, 31, 'Tanh-Sinh'),
        (15, 2, 6, 41, 'Tanh-Sinh'),
        (15, 3, 7, 41, 'Tanh-Sinh'),
        (25, 4, 9, 51, 'Tanh-Sinh'),
        (25, 5, 12, 61, 'Tanh-Sinh'),
        (25, 6, 15, 61, 'Tanh-Sinh'),
        (35, 8, 16, 81, 'Tanh-Sinh'),
        (51, 8, 24, 101, 'Tanh-Sinh'),
        (65, 8, 32, 101, 'Tanh-Sinh'),
    ]
    test_case(K, S, r, q, sigma, T, eta, parameter_sets)
