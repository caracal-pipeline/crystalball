import dask.array as da
import numpy as np

from daskms import Dataset
from Crystalball.filtering import valid_field_ids


def test_select_fields():
    datasets = []

    for field_name in ["PKS-1934", "3C286", "DEEP2"]:
        name = np.asarray(field_name, dtype=np.object)
        ds = Dataset({"NAME": (("row",), da.from_array(name, chunks=1))})
        datasets.append(ds)

    # No field selection, all fields returned
    assert valid_field_ids(datasets, None) == [0, 1, 2]

    assert valid_field_ids(datasets, "PKS-1934") == [0]
    assert valid_field_ids(datasets, "3C286, DEEP2") == [1, 2]
    assert valid_field_ids(datasets, "1, DEEP2") == [1, 2]
    assert valid_field_ids(datasets, "0, 1, 2") == [0, 1, 2]
    assert valid_field_ids(datasets, "2, 3") == [2]
