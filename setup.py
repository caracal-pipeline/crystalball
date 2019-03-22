#!/usr/bin/env python

import os
from distutils.core import setup

requirements = [
"codex-africanus[complete]>=0.1.4",
"regions"
]

PACKAGE_NAME = 'crystalball'
__version__ = '0.1.2'

setup(name = PACKAGE_NAME,
    version = __version__,
    description = "Predicts visibilities from a parameterised sky model",
    author = "Paolo Serra",
    author_email = "paolo80serra@gmail.com",
    url = "https://github.com/paoloserra/crystalball",
    packages=["Crystalball"], 
    install_requires = requirements,
    include_package_data = True,
    scripts = ["bin/" + i for i in os.listdir("bin")],
    license=["GNU GPL v2"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Astronomy"
    ]
     )
