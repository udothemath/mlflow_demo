import os
from mlflow import log_metric, log_param, log_artifact
import time
import mlflow

from functools import wraps

def elapsed_time(func):
    @wraps(func)
    def out(*args, **kwargs):
        init_time = time.time()
        func(*args, **kwargs)
        elapsed_time = time.time() - init_time
        print(f"Elapsed time of {func.__name__}: {elapsed_time:.4f}")
    return out

@elapsed_time
def main_func():
    # Log a parameter (key-value pair)
    log_param("param1", 5)

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", 1)
    log_metric("foo", 2)
    log_metric("foo", 3)

    # Log an artifact (output file)
    filename = "output_v2.txt"
    with open(filename, "w") as f:
        f.write("Hello world!")
    log_artifact(filename)

if __name__ == "__main__":
    # main_func()
    print( mlflow.tracking.get_tracking_uri())
    