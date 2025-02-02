import sys
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Determine Python version
python_version = sys.version_info

# Choose the correct C++ source file based on Python version
if python_version >= (3, 13):
    cpp_file = "fast_mssql_py_3-13.cpp"
elif python_version >= (3, 12):
    cpp_file = "fast_mssql_py_3-12.cpp"
else:
    raise RuntimeError("This package requires Python 3.12 or higher.")

ext_modules = [
    Pybind11Extension(
        "fast_mssql",
        [cpp_file],
        libraries=["odbc"],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="fast_mssql",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)