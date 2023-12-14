import multiprocessing
import time


def worker_function(args):
    x, y, z, w, u = args
    result = x * y + z - w + u
    time.sleep(u)
    return result


if __name__ == "__main__":
    with multiprocessing.Pool() as pool:
        # Submit multiple tasks to the pool with different inputs
        inputs = [(3, 4, 5, 6, 7), (1, 2, 3, 4, 5)]
        results = pool.apply_async(worker_function, inputs)

        # Wait for all tasks to complete
        results.wait()

        # Retrieve the results
        final_results = results.get()
        print("Final Results:", final_results)
