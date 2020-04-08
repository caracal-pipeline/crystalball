import dask.array as da
import numpy as np
import pytest

from daskms import Dataset
from crystalball.filtering import select_field_id


def test_select_field():
    datasets = []

    for field_name in ["DEEP2"]:
        name = np.asarray([field_name], dtype=np.object)
        ds = Dataset({"NAME": (("row",), da.from_array(name, chunks=1))})
        datasets.append(ds)

    # No field selection, single field id returned
    assert select_field_id(datasets, None) == 0

    datasets = []

    for field_name in ["PKS-1934", "3C286", "DEEP2"]:
        name = np.asarray([field_name], dtype=np.object)
        ds = Dataset({"NAME": (("row",), da.from_array(name, chunks=1))})
        datasets.append(ds)

    # No field selection, ValueError raised
    with pytest.raises(ValueError):
        assert select_field_id(datasets, None) == [0, 1, 2]

    assert select_field_id(datasets, "PKS-1934") == 0
    assert select_field_id(datasets, "0") == 0
    assert select_field_id(datasets, "3C286") == 1
    assert select_field_id(datasets, "1") == 1
    assert select_field_id(datasets, "DEEP2") == 2
    assert select_field_id(datasets, "2") == 2
