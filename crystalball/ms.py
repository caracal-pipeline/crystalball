# -*- coding: utf-8 -*-

import logging

import pyrap.tables as pt

log = logging.getLogger(__name__)


def ms_preprocess(args):
    """
    Adds output column if missing.

    Returns
    -------
    ms_rows : int
        number of Measurement Set Rows
    """

    # check output column
    with pt.table(args.ms, readonly=False) as ms:
        # Return if nothing todo
        if args.output_column in ms.colnames():
            return ms.nrows(), ms.coldatatype('DATA')

        log.info('inserting new column %s', args.output_column)
        desc = ms.getcoldesc("DATA")
        desc['name'] = args.output_column
        # python version hates spaces, who knows why
        desc['comment'] = desc['comment'].replace(" ", "_")
        dminfo = ms.getdminfo("DATA")
        dminfo["NAME"] = "%s-%s" % (dminfo["NAME"], args.output_column)
        ms.addcols(desc, dminfo)

        return ms.nrows(), ms.coldatatype('DATA')
