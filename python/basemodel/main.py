from pydantic import BaseModel

class A(BaseModel):
    a: int
    b: int

def func(a: A):
    a = A(**a)
    return a

print(func({"a": 1}))