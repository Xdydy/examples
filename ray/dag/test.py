import ray

@ray.remote
def f():
    return 1

@ray.remote
def g():
    a = 1
    b = f.bind()
    print(b)
    return a+b

print(g.bind())