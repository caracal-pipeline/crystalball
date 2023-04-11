#!/usr/bin/env python

from distutils.core import setup

requirements = [
    "codex-africanus[dask] >= 0.3.1",
    "dask-ms >= 0.2.6",
    "loguru",
    "regions >= 0.6",
    "psutil"
]

extras_require = {
  'testing': ['pytest', 'pytest-flake8']
}

PACKAGE_NAME = 'crystalball'
__version__ = '0.4.1'

setup(name=PACKAGE_NAME,
      version=__version__,
      description="Predicts visibilities from a parameterised sky model",
      author="Paolo Serra and others",
      author_email="paolo80serra@gmail.com",
      entry_points={
        'console_scripts': ['crystalball=crystalball.crystalball:predict'],
      },
      extras_require=extras_require,
      url="https://github.com/caracal-pipeline/crystalball",
      packages=["crystalball"],
      install_requires=requirements,
      include_package_data=True,
      python_requires=">=3.7",
      license="GNU GPL v2",
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering :: Astronomy"
      ])
