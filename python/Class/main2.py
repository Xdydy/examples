class MyClass:
    def __init__(self, value):
        self.m_value = value

    def __getattr__(self, name):
        """当尝试访问的属性或方法不存在时调用此方法"""
        attr = getattr(self.m_value, name)

        if callable(attr):
            def wrapper(*args, **kwargs):
                return attr(*args, **kwargs)
            return wrapper
        else:
            return attr
        
    def __add__(self, value):
        return self.m_value + value
    def __sub__(self,value):
        return self.m_value - value
    def __mul__(self,value):
        return self.m_value * value
    def __truediv__(self,value):
        return self.m_value / value
    def __iadd__(self,value):
        self.m_value += value
        return self
    def __isub__(self,value):
        self.m_value -= value
        return self
    def __imul__(self,value):
        self.m_value *= value
        return self
    def __itruediv__(self,value):
        self.m_value /= value
        return self


m = MyClass(4)
print(m+3)  # 输出 [1]
