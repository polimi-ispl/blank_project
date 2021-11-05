"""
Run a function using parallel processing

Blank Project

Image and Sound Processing Lab - Politecnico di Milano

Paolo Bestagini
"""
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm


def fun(x, y):
    """
    Simple function that sums two numbers
    :param x: first input number
    :param y: second input number
    :return: sum of the two numbers
    """
    return x+y


def main():
    # Input parameters
    x_list = np.arange(10)
    y = 5

    # Define the number of cores to use for parallel processing
    num_cpu = cpu_count() // 2  # Use one quarter of the available cores

    # We can only loop over one parameter (e.g., x), thus we need to fix the other function parameters (e.g., y=5)
    fun_part = partial(fun, y=y)

    # Initialize the pool of cores
    pool = Pool(num_cpu)

    # Evaluate the function in series
    t = time.time()
    result_series = []
    # for x in x_list:  # Without wait-bar
    for x in tqdm(x_list, total=len(x_list), desc='Serial'):  # With wait-bar
        result = fun_part(x)
        result_series.append(result)
    t_series = time.time() - t

    # Evaluate the function in parallel
    t = time.time()
    # result_parallel = pool.map(fun_part, x_list)  # Without wait-bar
    result_parallel = list(tqdm(pool.imap(fun_part, x_list), total=len(x_list), desc='Parallel'))  # With wait-bar
    t_parallel = time.time() - t

    # Close the pool
    pool.close()

    # Print results
    print('Serial results:   {} [{:.2f} ms]'.format(result_series, t_series*1000))
    print('Parallel results: {} [{:.2f} ms]'.format(result_parallel, t_parallel*1000))


if __name__ == '__main__':
    main()
