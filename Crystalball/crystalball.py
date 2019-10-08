#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from contextlib import ExitStack

from dask.diagnostics import ProgressBar
import numpy as np
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
from africanus.rime.dask import phase_delay, predict_vis
from africanus.model.coherency.dask import convert
from africanus.model.shape.dask import gaussian
from africanus.util.requirements import requires_optional

from Crystalball.budget import get_budget
from Crystalball.filtering import valid_field_ids, filter_datasets
from Crystalball.ms import ms_preprocess
from Crystalball.region import load_regions
from Crystalball.wsclean import import_from_wsclean


def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms",
                   help="Input .MS file.")
    p.add_argument("-sm", "--sky-model", default="sky-model.txt",
                   help="Name of file containing the sky model. "
                        "Default is 'sky-model.txt'")
    p.add_argument("-o", "--output-column", default="MODEL_DATA",
                   help="Output visibility column. Default is '%(default)s'")
    p.add_argument("-f", "--fields", type=str,
                   help="Comma-separated list of Field names or ids "
                        "which should be predicted. "
                        "All fields are predicted by default.")
    p.add_argument("-rc", "--row-chunks", type=int, default=0,
                   help="Number of rows of input .MS that are processed in "
                        "a single chunk. If 0 it will be set automatically. "
                        "Default is 0.")
    p.add_argument("-mc", "--model-chunks", type=int, default=0,
                   help="Number of sky model components that are processed in "
                        "a single chunk. If 0 it wil be set automatically. "
                        "Default is 0.")
    p.add_argument("--exp-sign-convention", choices=['casa', 'thompson'],
                   default='casa',
                   help="Sign convention to use for the complex exponential. "
                        "'casa' specifies the e^(2.pi.I) convention while "
                        "'thompson' specifies the e^(-2.pi.I) convention in "
                        "the white book and Fourier analysis literature. "
                        "Defaults to '%(default)s'")
    p.add_argument("-sp", "--spectra", action="store_true",
                   help="Optional. Model sources as non-flat spectra. "
                        "The spectral coefficients and reference frequency "
                        "must be present in the sky model.")
    p.add_argument("-w", "--within", type=str,
                   help="Optional. Give JS9 region file. Only sources within "
                        "those regions will be included.")
    p.add_argument("-po", "--points-only", action="store_true",
                   help="Select only point-type sources.")
    p.add_argument("-ns", "--num-sources", type=int, default=0, metavar="N",
                   help="Select only N brightest sources.")
    p.add_argument("-j", "--num-workers", type=int, default=0, metavar="N",
                   help="Explicitly set the number of worker threads.")
    p.add_argument("-mf", "--memory-fraction", type=float, default=0.5,
                   help="Fraction of system RAM that can be used. "
                        "Used when setting automatically the "
                        "chunk size. Default in 0.5.")

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


def corr_schema(pol):
    """
    Parameters
    ----------
    pol : :class:`xarray.Dataset`

    Returns
    -------
    corr_schema : list of list
        correlation schema from the POLARIZATION table,
        `[[9, 10], [11, 12]]` for example
    """

    corrs = pol.NUM_CORR.data[0]
    corr_types = pol.CORR_TYPE.data[0]

    if corrs == 4:
        return [[corr_types[0], corr_types[1]],
                [corr_types[2], corr_types[3]]]  # (2, 2) shape
    elif corrs == 2:
        return [corr_types[0], corr_types[1]]    # (2, ) shape
    elif corrs == 1:
        return [corr_types[0]]                   # (1, ) shape
    else:
        raise ValueError("corrs %d not in (1, 2, 4)" % corrs)


def einsum_schema(pol, dospec):
    """
    Returns an einsum schema suitable for multiplying per-baseline
    phase and brightness terms.

    Parameters
    ----------
    pol : :class:`xarray.Dataset`

    Returns
    -------
    einsum_schema : str
    """
    corrs = pol.NUM_CORR.values

    if corrs == 4:
        if dospec:
            return "srf, sfij -> srfij"
        else:
            return "srf, sij -> srfij"
    elif corrs in (2, 1):
        if dospec:
            return "srf, sfi -> srfi"
        else:
            return "srf, si -> srfi"
    else:
        raise ValueError("corrs %d not in (1, 2, 4)" % corrs)


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
    # get inclusion regions
    include_regions = load_regions(args.within) if args.within else []

    # Import source data from WSClean component list
    # See https://sourceforge.net/p/wsclean/wiki/ComponentList
    (comp_type, radec, stokes,
     spec_coeff, ref_freq, log_spec_ind,
     gaussian_shape) = import_from_wsclean(args.sky_model,
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
    args.row_chunks, args.model_chunks = get_budget(comp_type.shape[0],
                                                    ms_rows,
                                                    max_num_chan,
                                                    max_num_corr,
                                                    ms_datatype, args)

    radec = da.from_array(radec, chunks=(args.model_chunks, 2))
    stokes = da.from_array(stokes, chunks=(args.model_chunks, 4))

    if np.count_nonzero(comp_type == 'GAUSSIAN') > 0:
        gaussian_components = True
        gshape_chunks = (args.model_chunks, 3)
        gaussian_shape = da.from_array(gaussian_shape, chunks=gshape_chunks)
    else:
        gaussian_components = False

    if args.spectra:
        spec_chunks = (args.model_chunks, spec_coeff.shape[1])
        spec_coeff = da.from_array(spec_coeff, chunks=spec_chunks)
        ref_freq = da.from_array(ref_freq, chunks=(args.model_chunks,))

    # List of write operations
    writes = []

    # Construct a graph for each FIELD and DATA DESCRIPTOR
    datasets = xds_from_ms(args.ms,
                           columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
                           group_cols=["FIELD_ID", "DATA_DESC_ID"],
                           chunks={"row": args.row_chunks})

    select_fields = valid_field_ids(field_ds, args.fields)

    for xds in filter_datasets(datasets, select_fields):
        # Extract frequencies from the spectral window associated
        # with this data descriptor id
        field = field_ds[xds.attrs['FIELD_ID']]
        ddid = ddid_ds[xds.attrs['DATA_DESC_ID']]
        spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.data[0]]
        pol = pol_ds[ddid.POLARIZATION_ID.data[0]]
        frequency = spw.CHAN_FREQ.data[0]

        corrs = pol.NUM_CORR.values

        lm = radec_to_lm(radec, field.PHASE_DIR.data[0][0])

        if args.exp_sign_convention == 'casa':
            uvw = -xds.UVW.data
        elif args.exp_sign_convention == 'thompson':
            uvw = xds.UVW.data
        else:
            raise ValueError("Invalid sign convention '%s'" % args.sign)

        if args.spectra:
            # flux density at reference frequency ...
            # ... for logarithmic polynomial functions
            if log_spec_ind:
                Is = da.log(stokes[:, 0, None])*frequency[None, :]**0
            # ... or for ordinary polynomial functions
            else:
                Is = stokes[:, 0, None]*frequency[None, :]**0
            # additional terms of SED ...
            for jj in range(spec_coeff.shape[1]):
                # ... for logarithmic polynomial functions
                if log_spec_ind:
                    Is += spec_coeff[:, jj, None] * \
                        da.log((frequency[None, :]/ref_freq[:, None])**(jj+1))
                # ... or for ordinary polynomial functions
                else:
                    Is += spec_coeff[:, jj, None] * \
                        (frequency[None, :]/ref_freq[:, None]-1)**(jj+1)
            if log_spec_ind:
                Is = da.exp(Is)
            Qs = da.zeros_like(Is)
            Us = da.zeros_like(Is)
            Vs = da.zeros_like(Is)
            # stack along new axis and make it the last axis of the new array
            spectrum = da.stack([Is, Qs, Us, Vs], axis=-1)
            spectrum = spectrum.rechunk(
                spectrum.chunks[:2] + (spectrum.shape[2],))

        print('-------------------------------------------')
        print('Nr sources        = {0:d}'.format(stokes.shape[0]))
        print('-------------------------------------------')
        print('stokes.shape      = {0:}'.format(stokes.shape))
        print('frequency.shape   = {0:}'.format(frequency.shape))
        if args.spectra:
            print('Is.shape          = {0:}'.format(Is.shape))
        if args.spectra:
            print('spectrum.shape    = {0:}'.format(spectrum.shape))

        # (source, row, frequency)
        phase = phase_delay(lm, uvw, frequency)
        # If at least one Gaussian component is present in the component
        # list then all sources are modelled as Gaussian components
        # (Delta components have zero width)
        if gaussian_components:
            phase *= gaussian(uvw, frequency, gaussian_shape)
        # (source, frequency, corr_products)
        brightness = convert(spectrum if args.spectra else stokes,
                             ["I", "Q", "U", "V"],
                             corr_schema(pol))

        print('brightness.shape  = {0:}'.format(brightness.shape))
        print('phase.shape       = {0:}'.format(phase.shape))
        print('-------------------------------------------')
        print('Attempting phase-brightness einsum with "{0:s}"'
              .format(einsum_schema(pol, args.spectra)))

        # (source, row, frequency, corr_products)
        jones = da.einsum(einsum_schema(pol, args.spectra), phase, brightness)
        print('jones.shape       = {0:}'.format(jones.shape))
        print('-------------------------------------------')
        if gaussian_components:
            print('Some Gaussian sources found')
        else:
            print('All sources are Delta functions')
        print('-------------------------------------------')

        # Identify time indices
        _, time_index = da.unique(xds.TIME.data, return_inverse=True)

        # Predict visibilities
        vis = predict_vis(time_index, xds.ANTENNA1.data, xds.ANTENNA2.data,
                          None, jones, None, None, None, None)

        # Reshape (2, 2) correlation to shape (4,)
        if corrs == 4:
            vis = vis.reshape(vis.shape[:2] + (4,))

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
            stack.enter_context(ProgressBar())
        else:
            # Log progress every 5 minutes
            stack.enter_context(ProgressBar(minimum=2*60, dt=5))

        # Submit all graph computations in parallel
        dask.compute(writes)
