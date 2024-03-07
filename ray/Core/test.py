def func():
    print("func() called")
    ...

def func(a):
    print("func(a) called")
    ...

def func(a, b):
    print("func(a, b) called")
    ...

def func(*args):
    for arg in args:
        print(arg)

func(1)
func(1,2)
func(1,2,3)

