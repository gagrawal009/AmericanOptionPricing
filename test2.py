from utils.CompleteAlgorithm import main
import time

# Fix algorithm parameters, test option parameters S, r, q, T.
def test_case(K, sigma, eta, parameter_sets):
    """
    Test the performance and accuracy of the `main` function with different parameter sets.
    Fix l=131, m=16, n=64, p=131, method='Tanh-Sinh'.

    Parameters:
        K (float): Strike price.
        sigma (float): Volatility.
        eta (float): Relaxation parameter for iteration.
        parameter_sets (list of tuples): A list of parameter sets to test.
            Each tuple should contain the following values:
                - S (float): Current stock price.
                - r (float): Risk-free interest rate.
                - q (float): Dividend yield.
                - T (float): Time to maturity.
    Outputs:
        Prints:
            - The computed American premium for each parameter set.
            - The CPU time taken for each computation.

    Returns:
        None: Results are printed to the console.
    """
    for params in parameter_sets:
        l, m, n, p, method = (131, 16, 64, 131, 'Tanh-Sinh')
        S, r, q, T = params

        start_time = time.perf_counter()
        V = main(
            K=K, S=S, r=r, q=q, sigma=sigma, tau_max=T,
            l=l, m=m, n=n, p=p, eta=eta, method=method
        )
        end_time = time.perf_counter()
        print(f"S={S}, r={r}, q={q}, T={T}")
        print("American Premium: {:.12f}".format(V))
        elapsed_time = end_time - start_time
        print(f"CPU Seconds: {elapsed_time:.2E}")


if __name__ == '__main__':
    K = 100.
    sigma = 0.25
    eta = 0.01
    parameter_sets = [
        (100, 0.05, 0.05, 1),
        # Add more sets.
    ]
    test_case(K, sigma, eta, parameter_sets)
