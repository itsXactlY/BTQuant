import sys
import os
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

python_version = sys.version_info
if python_version >= (3, 13):
    cpp_file = "fast_mssql_py_3-13.cpp"
elif python_version >= (3, 12):
    cpp_file = "fast_mssql_py_3-12.cpp"
else:
    raise RuntimeError("This package requires Python 3.12 or higher.")

if os.name == "nt":
    include_dirs = [
        r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\um",
        r"C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\shared",
    ]
    library_dirs = [
        r"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.19041.0\um",
    ]
    libraries = ["odbc32"]
    extra_compile_args = ["/std:c++latest", "/EHsc", "/bigobj"]
    extra_link_args = []
else:
    include_dirs = []
    library_dirs = []
    libraries = ["odbc"]
    extra_compile_args = ["-O3"]
    extra_link_args = []

ext_modules = [
    Pybind11Extension(
        "fastquant.data.mssql.MsSQL.fast_mssql",
        [cpp_file],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="fast_mssql",
    version="1.0",
    description="A fast MSSQL extension for fastquant",
    ext_modules=ext_modules,
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
)