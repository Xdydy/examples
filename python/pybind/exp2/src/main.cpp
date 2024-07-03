#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

class MyClass {
private:
    int i;
public:
    MyClass(int i) : i(i) {}
    int get_i() const { return i; }
};

namespace py=pybind11;

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function which adds two numbers");

    py::class_<MyClass>(m, "MyClass")
       .def(py::init<int>())
       .def("get_i", &MyClass::get_i);
}