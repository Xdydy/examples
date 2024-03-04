def log():
    print('log')
    def decorator(func):
        print('decorator')
        def wrapper(*args, **kwargs):
            print('wrapper')
            return func(*args, **kwargs)
        return wrapper
    return decorator

@log()
def f(x):
    return x*x

print(f(3))  # 9