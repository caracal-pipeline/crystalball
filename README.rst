Reference paper
=============

Serra, Perkins & Smirnov (2022), ASPC, 532, 409

ADS: https://ui.adsabs.harvard.edu/abs/2022ASPC..532..409S

BibTex:

..  code-block:: bash

    @INPROCEEDINGS{2022ASPC..532..409S,
      author = {{Serra}, Paolo and {Perkins}, Simon and {Smirnov}, Oleg},
      title = "{CRYSTALBALL: A Numba and Dask-Accelerated Fourier Transform of a Sky Model into an Interferometry Dataset}",
      booktitle = {Astronomical Data Analysis Software and Systems XXX},
      year = 2022,
      editor = {{Ruiz}, Jose Enrique and {Pierfedereci}, Francesco and {Teuben}, Peter},
      series = {Astronomical Society of the Pacific Conference Series},
      volume = {532},
      month = jul,
      pages = {409},
      adsurl = {https://ui.adsabs.harvard.edu/abs/2022ASPC..532..409S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


Applicability
=============

Crystalball currently supports prediction from a WSClean list of delta and Gaussian components with (log-) polynomial spectral shape (https://sourceforge.net/p/wsclean/wiki/ComponentList) into the MODEL_DATA column of a measurement set. This is done largely based on code available in the https://github.com/ska-sa/codex-africanus library.

Installation in a virtual environment
=====================================

Crystalball depends on python-casacore which builds from source.
The dependencies mentioned at the following links must be installed
in order for the build to succeed:

- https://github.com/casacore/casacore#building-from-source
- https://github.com/casacore/python-casacore#from-source

Create and activate a Python3 virtual environment (python 3.10 or higher)

``virtualenv -p python3 <name-of-virtualenv>``

(systems without a Python2 installation don't even need the ``-p python3`` specifier, but it doesn't hurt.)

``source <name-of-virtualenv>/bin/activate``

Pip install crystalball

``pip install <path to crystalball>``

Run crystalball
===============

Activate the virtual environment where you installed codex-africanus, xarray and xarray-ms (see above)

``source <name-of-virtualenv>/bin/activate``

Run crystalball

``crystalball <file.ms> [-h] [-sm SKY_MODEL] [-rc ROW_CHUNKS] [-mc MODEL_CHUNKS] [-f FIELD] [-mf MEMORY_FRACTION] [-w REGION_FILE] [-po] [-ns NUM_BRIGHTEST_SOURCES] [-j NUM_WORKERS] [-o OUTPUT_COLUMN]``
