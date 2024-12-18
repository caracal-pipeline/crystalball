# -*- coding: utf-8 -*-
from __future__ import annotations

import logging

import casacore.tables as pt

log = logging.getLogger(__name__)


def ms_preprocess(
        ms_name: str,
        output_column: str = "CORRECTED_DATA",
):
    """
    Adds output column if missing.

    Returns
    -------
    ms_rows : int
        number of Measurement Set Rows
    """

    # check output column
    with pt.table(ms_name, readonly=False) as ms:
        # Return if nothing todo
        if output_column in ms.colnames():
            return ms.nrows(), ms.coldatatype('DATA')

        log.info('inserting new column %s', output_column)
        desc = ms.getcoldesc("DATA")
        desc['name'] = output_column
        # python version hates spaces, who knows why
        desc['comment'] = desc['comment'].replace(" ", "_")
        dminfo = ms.getdminfo("DATA")
        dminfo["NAME"] = "%s-%s" % (dminfo["NAME"], output_column)
        ms.addcols(desc, dminfo)

        return ms.nrows(), ms.coldatatype('DATA')
