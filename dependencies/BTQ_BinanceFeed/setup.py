from setuptools import setup, find_packages

setup(
    name='BTQ_Binance',
    version='0.0.1.3',
    description="Initial inspired by a fork of Ed Bartosh's CCXT Store Work with some additions",
    # url='https://github.com/Dave-Vallance/bt-ccxt-store',
    author='ItsXactlY',
    author_email='no',
    license='MIT',
    packages=find_packages(),  # Automatically find the package directory
    install_requires=['backtrader', 'ipython'],
)