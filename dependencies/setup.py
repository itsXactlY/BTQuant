#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
###############################################################################
#
# Copyright (C) 2015-2023 Daniel Rodriguez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
###############################################################################
import os.path
import codecs  # To use a consistent encoding
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the relevant file
with codecs.open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# Package name
pname = 'backtrader'

# Get the version ... execfile is only on Py2 ... use exec + compile + open
vname = 'version.py'
with open(os.path.join(pname, vname)) as f:
    exec(compile(f.read(), vname, 'exec'))

# Generate links
gurl = 'https://github.com/ItsXactly/' + pname
gdurl = gurl + '/tarball/' + __version__

setuptools.setup(
    name=pname,

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=__version__,

    description='BackTesting Engine',
    long_description=long_description,

    # The project's main homepage.
    url=gurl,
    download_url=gdurl,

    # Author details
    author='Daniel Rodriguez',
    author_email='danjrod@gmail.com',

    # Choose your license
    license='GPLv3+',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',

        # Indicate which Topics are covered by the package
        'Topic :: Software Development',
        'Topic :: Office/Business :: Financial',

        # Pick your license as you wish (should match "license" above)
        ('License :: OSI Approved :: ' +
         'GNU General Public License v3 or later (GPLv3+)'),
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: OS Independent',
    ],

    # What does your project relate to?
    keywords=['trading', 'development'],
    packages=setuptools.find_packages(exclude=['docs', 'docs2', 'samples']),
    package_data={
        "backtrader": ["feeds/mssql/*"],
    },
    include_package_data=True,

    install_requires=[
        'ccxt',
        'pybind11',
        'pyodbc',
        'websockets',
        'websocket-client==1.8.0',
        'Web3',
        'matplotlib',
        'pandas',
        'numpy',
        'polars',
        'pyarrow',
        'telethon',
        'scikit-learn',
        'keras',
        'pytz'
    ],

    extras_require={
        'plotting':  ['matplotlib'],
    },

    entry_points={'console_scripts': ['btrun=backtrader.btrun:btrun']},

    scripts=['tools/bt-run.py'],
)