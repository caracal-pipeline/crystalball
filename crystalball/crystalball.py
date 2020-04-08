#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from contextlib import ExitStack
import warnings

from dask.array import PerformanceWarning
from loguru import logger as log
import sys


try:
    import dask
    import dask.array as da
    from daskms import xds_from_ms, xds_from_table, xds_to_table
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None

from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import wsclean_predict
from africanus.util.dask_util import EstimatingProgressBar
from africanus.util.requirements import requires_optional

import crystalball.logger_init  # noqa
from crystalball.budget import get_budget
from crystalball.filtering import select_field_id, filter_datasets
from crystalball.ms import ms_preprocess
from crystalball.region import load_regions
from crystalball.wsclean import import_from_wsclean, WSCleanModel


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms",
                   help="Input .MS file.")
    p.add_argument("-sm", "--sky-model", default="sky-model.txt",
                   help="Name of file containing the sky model. "
                        "Default is 'sky-model.txt'")
    p.add_argument("-o", "--output-column", default="MODEL_DATA",
                   help="Output visibility column. Default is '%(default)s'")
    p.add_argument("-f", "--field", type=str,
                   help="The field name or id to be predicted. "
                         "If not provided, only a single field "
                         "may be present in the MS")
    p.add_argument("-rc", "--row-chunks", type=int, default=0,
                   help="Number of rows of input MS that are processed in "
                        "a single chunk. If 0 it will be set automatically. "
                        "Default is 0.")
    p.add_argument("-mc", "--model-chunks", type=int, default=0,
                   help="Number of sky model components that are processed in "
                        "a single chunk. If 0 it wil be set automatically. "
                        "Default is 0.")
    p.add_argument("-w", "--within", type=str,
                   help="Optional. Give JS9 region file. Only sources within "
                        "those regions will be included.")
    p.add_argument("-po", "--points-only", action="store_true",
                   help="Select only point-type sources.")
    p.add_argument("-ns", "--num-sources", type=int, default=0, metavar="N",
                   help="Select only N brightest sources.")
    p.add_argument("-j", "--num-workers", type=int, default=0, metavar="N",
                   help="Explicitly set the number of worker threads.")
    p.add_argument("-mf", "--memory-fraction", type=float, default=0.1,
                   help="Fraction of system RAM that can be used. "
                        "Used when setting automatically the "
                        "chunk size. Default in 0.1.")

    return p


def support_tables(args, tables):
    """
    Parameters
    ----------
    args : object
        Script argument objects
    tables : list of str
        List of support tables to open

    Returns
    -------
    table_map : dict of :class:`xarray.Dataset`
        {name: dataset}
    """
    return {t: [ds.compute() for ds in
                xds_from_table("::".join((args.ms, t)),
                               group_cols="__row__")]
            for t in tables}


def fill_correlations(vis, pol):
    """
    Expands single correlation produced by wsclean_predict to the
    full set of correlations.

    Parameters
    ----------
    vis : :class:`dask.array.Array`
        dask array of visibilities of shape :code:`(row, chan, 1)`
    pol : :class:`xarray.Dataset`
        MS Polarisation dataset.

    Returns
    -------
    vis : :class:`dask.array.Array`
        dask array of visibilities of shape :code:`(row, chan, corr)`
    """

    corrs = pol.NUM_CORR.data[0]

    assert vis.ndim == 3

    if corrs == 1:
        return vis
    elif corrs == 2:
        vis = da.concatenate([vis, vis], axis=2)
        return vis.rechunk({2: corrs})
    elif corrs == 4:
        zeros = da.zeros_like(vis)
        vis = da.concatenate([vis, zeros, zeros, vis], axis=2)
        return vis.rechunk({2: corrs})
    else:
        raise ValueError("MS Correlations %d not in (1, 2, 4)" % corrs)


def source_model_to_dask(source_model, chunks):
    # Create chunked dask arrays from wsclean model arrays
    sm = source_model

    radec_chunks = (chunks,) + sm.radec.shape[1:]
    spi_chunks = (chunks,) + sm.spi.shape[1:]
    gauss_chunks = (chunks,) + sm.gauss_shape.shape[1:]

    return WSCleanModel(da.from_array(sm.source_type, chunks=chunks),
                        da.from_array(sm.radec, chunks=radec_chunks),
                        da.from_array(sm.flux, chunks=chunks),
                        da.from_array(sm.spi, chunks=spi_chunks),
                        da.from_array(sm.ref_freq, chunks=chunks),
                        da.from_array(sm.log_poly, chunks=chunks),
                        da.from_array(sm.gauss_shape, chunks=gauss_chunks))


def predict():
    # Parse application args
    args = create_parser().parse_args([a for a in sys.argv[1:]])

    with ExitStack() as stack:
        # Set up dask ThreadPool prior to any application dask calls
        if args.num_workers:
            stack.enter_context(dask.config.set(num_workers=args.num_workers))

        # Run application script
        return _predict(args)


@requires_optional("dask.array", "daskms", opt_import_error)
def _predict(args):
    import pkg_resources
    version = pkg_resources.get_distribution("crystalball").version
    log.info("Crystalball version {0}", version)

    # get inclusion regions
    include_regions = load_regions(args.within) if args.within else []

    # Import source data from WSClean component list
    # See https://sourceforge.net/p/wsclean/wiki/ComponentList
    source_model = import_from_wsclean(args.sky_model,
                                       include_regions=include_regions,
                                       point_only=args.points_only,
                                       num=args.num_sources or None)

    # Add output column if it isn't present
    ms_rows, ms_datatype = ms_preprocess(args)

    # Get the support tables
    tables = support_tables(args, ["FIELD", "DATA_DESCRIPTION",
                                   "SPECTRAL_WINDOW", "POLARIZATION"])

    field_ds = tables["FIELD"]
    ddid_ds = tables["DATA_DESCRIPTION"]
    spw_ds = tables["SPECTRAL_WINDOW"]
    pol_ds = tables["POLARIZATION"]

    max_num_chan = max([ss.NUM_CHAN.data[0] for ss in spw_ds])
    max_num_corr = max([ss.NUM_CORR.data[0] for ss in pol_ds])

    # Perform resource budgeting
    nsources = source_model.source_type.shape[0]
    args.row_chunks, args.model_chunks = get_budget(nsources,
                                                    ms_rows,
                                                    max_num_chan,
                                                    max_num_corr,
                                                    ms_datatype, args)

    source_model = source_model_to_dask(source_model, args.model_chunks)

    # List of write operations
    writes = []

    datasets = xds_from_ms(args.ms,
                           columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
                           group_cols=["FIELD_ID", "DATA_DESC_ID"],
                           chunks={"row": args.row_chunks})

    field_id = select_field_id(field_ds, args.field)

    for xds in filter_datasets(datasets, field_id):
        # Extract frequencies from the spectral window associated
        # with this data descriptor id
        field = field_ds[xds.attrs['FIELD_ID']]
        ddid = ddid_ds[xds.attrs['DATA_DESC_ID']]
        spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.data[0]]
        pol = pol_ds[ddid.POLARIZATION_ID.data[0]]
        frequency = spw.CHAN_FREQ.data[0]

        corrs = pol.NUM_CORR.values

        lm = radec_to_lm(source_model.radec, field.PHASE_DIR.data[0][0])

        with warnings.catch_warnings():
            # Ignore dask chunk warnings emitted when going from 1D
            # inputs to a 2D space of chunks
            warnings.simplefilter('ignore', category=PerformanceWarning)
            vis = wsclean_predict(xds.UVW.data,
                                  lm,
                                  source_model.source_type,
                                  source_model.flux,
                                  source_model.spi,
                                  source_model.log_poly,
                                  source_model.ref_freq,
                                  source_model.gauss_shape,
                                  frequency)

        vis = fill_correlations(vis, pol)

        log.info('Field {0} DDID {1:d} rows {2} chans {3} corrs {4}',
                 field.NAME.values[0],
                 xds.DATA_DESC_ID,
                 vis.shape[0], vis.shape[1], vis.shape[2])

        # Assign visibilities to MODEL_DATA array on the dataset
        xds = xds.assign(
            **{args.output_column: (("row", "chan", "corr"), vis)})
        # Create a write to the table
        write = xds_to_table(xds, args.ms, [args.output_column])
        # Add to the list of writes
        writes.append(write)

    with ExitStack() as stack:
        if sys.stdout.isatty():
            # Default progress bar in user terminal
            stack.enter_context(EstimatingProgressBar())
        else:
            # Log progress every 5 minutes
            stack.enter_context(EstimatingProgressBar(minimum=2 * 60, dt=5))

        # Submit all graph computations in parallel
        dask.compute(writes)
