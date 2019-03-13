# Applicability

Currently supports prediction from a WSClean list of delta and Gaussian components with (log-) polynomial spectral shape (https://sourceforge.net/p/wsclean/wiki/ComponentList) into the MODEL_DATA column of a measurement set.

# Installation in a virtual environment

Get crystalball and the required codex-africanus (by Simon Perkins)
```
git clone https://github.com/paoloserra/crystalball
git clone https://github.com/ska-sa/codex-africanus
```
Create and activate a virtual environment
```
virtualenv <name-of-virtualenv>
source <name-of-virtualenv>/bin/activate
```
Pip install codex-africanus and a few other dependencies
```
cd codex-africanus
pip install -e .[dask,scipy,astropy,python-casacore]
pip install xarray
pip install xarray-ms
```
Pip install crystalball
```
pip install <path to crystalball>
```

# Run crystalball

Activate the virtual environment where you installed codex-africanus, xarray and xarray-ms (see above)
```
source <name-of-virtualenv>/bin/activate
```
Run crystalball
```
crystalball <file.ms> [-h] [-sm SKY_MODEL] [-rc ROW_CHUNKS] [-mc MODEL_CHUNKS] [-iuvw] [-sp]
```
