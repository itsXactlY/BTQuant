from setuptools import setup, find_packages

setup(
    name='BTQuant_Exchange_Adapters',
    version='0.0.1.5',
    description="Initial inspired by a fork of Ed Bartosh's CCXT Store. Reworked to arrive in 2024.",
    # url='https://github.com/Dave-Vallance/bt-ccxt-store',
    author='ItsXactlY',
    author_email='no',
    license='MIT',
    packages=find_packages(),  # Automatically find the package directory
    install_requires=['backtrader', 'ipython', 'websockets'],
)