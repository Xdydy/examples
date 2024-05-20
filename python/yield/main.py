def lt(a,b):
    yield a < b

class A:
    def __init__(self,value) -> None:
        self.value = value
        pass
    def __lt__(self,other):
        print("lt")
        return lt(self.value,other.value)
    
        

def f():
    a = A(1)
    b = A(2)
    if a < b:
        print("a < b")
    else:
        print("a >= b")

f()

def foo():
    yield 1

print(next(foo()))
print(next(foo()))

g = lambda x: lambda y: x + y

g = g(1)
print(g(1))


def g(a,b):
    return a+b
def f():
    res = yield g
    print(res)
    yield res

gen = f()
res1 = next(gen)
print(res1)
res2 = res1(1,2)
print(res2)
res3 = gen.send(res2)
print(res3)
