import ray


@ray.remote
class Actor:
    def __init__(self, init_value):
        self.i = init_value

    def inc(self, x):
        self.i += x

    def get(self):
        return self.i

@ray.remote
def combine(x,y):
    a1 = Actor.bind(10)
    a1.get.bind()
    return x+y

a1 = Actor.bind(10)
res = a1.get.bind()
# print(res)
# assert ray.get(res.execute()) == 10

a2 = Actor.bind(10)
a1.inc.bind(2)
a1.inc.bind(4)
a2.inc.bind(6)
dag = combine.bind(a1.get.bind(), a2.get.bind())

print(dag)
# assert ray.get(dag.execute()) == 32
