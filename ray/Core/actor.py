import ray
ray.init()

@ray.remote
class Counter(object):
    def __init__(self):
        self.value = 0

    def increment(self):
        self.value += 1
        return self.value

    def read(self):
        return self.value
    
counters = [Counter.remote() for _ in range(4)]
results = [c.increment.remote() for c in counters]
futures = [c.read.remote() for c in counters]
print(ray.get(futures))  # [1, 1, 1, 1]