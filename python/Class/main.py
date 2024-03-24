class A:
    def __init__(self,a) -> None:
        self.a = a
        pass

class B(A):
    def __init__(self,a) -> None:
        super().__init__(a)
        self.b = a

class As:
    def __init__(self) -> None:
        self.set_a(A)
        pass
    def set_a(self, cls:A):
        self.a = cls(1)

class Bs(As):
    def __init__(self) -> None:
        super().__init__()
        self.set_a(B)
        pass
    def set_a(self, cls):
        super().set_a(cls)

if __name__ == "__main__":
    bs = Bs()
    print(bs.a.a)
    print(bs.a.b)