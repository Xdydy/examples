from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension("example", ["src/main.cpp"]),
]

setup(
    name="example",
    ext_modules=ext_modules,
    build_ext=build_ext
)