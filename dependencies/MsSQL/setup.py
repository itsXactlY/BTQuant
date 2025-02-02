import sys
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Determine Python version and choose the correct C++ source file
python_version = sys.version_info
if python_version >= (3, 13):
    cpp_file = "fast_mssql_py_3-13.cpp"
elif python_version >= (3, 12):
    cpp_file = "fast_mssql_py_3-12.cpp"
else:
    raise RuntimeError("This package requires Python 3.12 or higher.")

ext_modules = [
    Pybind11Extension(
        # Install inside the fastquant package structure:
        "fastquant.data.mssql.MsSQL.fast_mssql",
        [cpp_file],
        libraries=["odbc"],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="fast_mssql",
    version="0.1",
    description="A fast MSSQL extension for fastquant",
    ext_modules=ext_modules,
    packages=find_packages(),  # This will find your fastquant package
    cmdclass={"build_ext": build_ext},
)
