from setuptools import setup
from setuptools.command.install import install
from setuptools.dist import Distribution

class BinaryDistribution(Distribution):
  def has_ext_modules(self):
    return True

class InstallPlatlib(install):
    def finalize_options(self):
        super().finalize_options()
        self.install_lib=self.install_platlib

setup(
  name='ccapi',
  version='1.3.3.7',
  distclass=BinaryDistribution,
  cmdclass={'install': InstallPlatlib},
  packages=[''],
  py_modules = ['ccapi'],
  package_data={'': ['_ccapi_binding_python.so']},
  install_requires=['setuptools', 'wheel'],
)
