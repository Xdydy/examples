#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(pyexample, m) {
    m.doc() = "This is an emample";
    m.def("add", &add, "This is an add func");
}