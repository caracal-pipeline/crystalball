import logging

log = logging.getLogger(__name__)


def valid_field_ids(field_datasets, fields=None):
    """
    Parameters
    ----------
    field_datasets : list of :class:`daskms.Dataset`

    fields : None or str
        if None, all fields are valid, otherwise
        the string will be split on commas and used to
        select fields. Both field ID's and names are supported
        in the string.

        .. code-block:: python

            valid_field_ids(field_ds, ["PKS-1934,1,DEEP2,3"])

    Returns
    -------
    fields_ids : list of ints
        List of valid field IDs
    """
    if fields is None:
        return list(range(len(field_datasets)))
    elif not isinstance(fields, str):
        raise TypeError("fields must be None or str")

    field_names = {}

    for i, ds in enumerate(field_datasets):
        names = ds.NAME.values
        assert len(names) == 1, "Should have a single name per row"
        field_names[str(names[0])] = i

    selected_fields = [f.strip() for f in fields.split(',')]

    # Add integral lookup values
    for field_name, field_id in list(field_names.items()):
        exists = field_names.setdefault(str(field_id), field_id)

        if not exists == field_id:
            raise ValueError("Existing field_id mismatch "
                             "for field %s. %d != %d"
                             % (field_name, exists, field_id))

    field_ids = []

    for f in selected_fields:
        try:
            field_id = field_names[f]
        except KeyError:
            log.warning("Field %s does not exist and ignored for selection", f)
        else:
            field_ids.append(field_id)

    return field_ids


def filter_datasets(datasets, field_ids):
    """
    Parameters
    ----------
    datasets : list of :class:`daskms.Dataset`
        List of datasets representing partitions of the Measurement Set.
        Should be partitioned by FIELD_ID and DATA_DESC_ID at least
    field_ids : list of ints
        List of valid fields

    Returns
    -------
    datasets : list of :class:`daskms.Dataset`
        Filtered list of datasets
    """
    field_ids = set(field_ids)
    return [ds for ds in datasets if ds.FIELD_ID in field_ids]
