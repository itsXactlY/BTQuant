from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "fast_mssql",
        ["fast_mssql.cpp"],
        libraries=["odbc"],
        extra_compile_args=["-O3"],
    ),
]

setup(
    name="fast_mssql",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)