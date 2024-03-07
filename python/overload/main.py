from typing import overload
from typing import Callable

class Function1:
    def __init__(self, function) -> None:
        print("Function1")
        pass

class Function2:
    def __init__(self, function) -> None:
        print("Function2")
        pass
    def remote(self, __arg0, __arg1):
        ...

class Function:
    def __init__(self, function) -> None:
        print("Function")
        self.remote = function
        pass

@overload
def remote(function: Callable[[any], None]) -> Function1:
    ...    

@overload
def remote(function: Callable[[any,any],None]) -> Function2:
    ...


def remote(*args) -> Function:
    return Function(args[0])


@remote
def f(x):
    return x*x

print(type(f))
result = f.remote(2)
print(result)