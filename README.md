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

# Run crystalball

```
python crystalball.py <file.ms> [-h] [-sm SKY_MODEL] [-rc ROW_CHUNKS] [-mc MODEL_CHUNKS] [-iuvw] [-sp]
```
