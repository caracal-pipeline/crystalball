import logging

log = logging.getLogger(__name__)


def select_field_id(field_datasets, field=None):
    """
    Parameters
    ----------
    field_datasets : list of :class:`daskms.Dataset`

    field : None or str
        if None, will try and return the only field in the MS.

        .. code-block:: python

            select_field_id(field_ds, "PKS-1934")

    Returns
    -------
    fields_id : int
        Selected field ID
    """
    if field is None:
        if len(field_datasets) == 1:
            return 0

        names = ["%s or %d" % (ds.NAME.values[0], i)
                 for i, ds in enumerate(field_datasets)]

        raise ValueError("No field was provided "
                         "but multiple fields are present "
                         "in the Measurement Set. "
                         "Please select a field %s."
                         % names)
    elif not isinstance(field, str):
        raise TypeError("field must be None or str")

    for i, ds in enumerate(field_datasets):
        names = ds.NAME.values
        assert len(names) == 1, "Should have a single name per row"

        if str(names[0]) == field or str(i) == field:
            return i

    names = ["%s or %d" % (ds.NAME.values[0], i)
             for i, ds in enumerate(field_datasets)]

    raise ValueError("%s was requested, but no matching field "
                     "was found %s" % (field, names))


def filter_datasets(datasets, field_id):
    """
    Parameters
    ----------
    datasets : list of :class:`daskms.Dataset`
        List of datasets representing partitions of the Measurement Set.
        Should be partitioned by FIELD_ID and DATA_DESC_ID at least
    field_id : list of ints
        List of valid fields

    Returns
    -------
    datasets : list of :class:`daskms.Dataset`
        Filtered list of datasets
    """
    return [ds for ds in datasets if ds.FIELD_ID == field_id]
