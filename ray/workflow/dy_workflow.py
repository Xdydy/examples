import ray
from ray import workflow
@ray.remote
def add(a: int, b: int) -> int:
    print(f"add: {a} + {b}")
    return a + b

@ray.remote
def fib(n: int) -> int:
    print(f"fib: {n}")
    if n <= 1:
        return n
    # return a continuation of a DAG
    return workflow.continuation(add.bind(fib.bind(n - 1), fib.bind(n - 2)))

# assert workflow.run(fib.bind(10)) == 55
print(workflow.run(fib.bind(10)))