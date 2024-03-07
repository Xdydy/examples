import ray

ray.init()

@ray.remote
def my_function():
    return 1

my_function.options(num_cpus=3).remote()