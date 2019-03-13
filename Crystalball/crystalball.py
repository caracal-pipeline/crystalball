#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

from dask.diagnostics import ProgressBar
import numpy as np
import sys


try:
    from astropy.coordinates import Angle
except ImportError as e:
    astropy_import_error = e
else:
    astropy_import_error = None

try:
    import dask
    import dask.array as da
    import xarray as xr
    from xarrayms import xds_from_ms, xds_from_table, xds_to_table
except ImportError as e:
    opt_import_error = e
else:
    opt_import_error = None

from africanus.coordinates.dask import radec_to_lm
from africanus.rime.dask import phase_delay, predict_vis
from africanus.model.coherency.dask import convert
from africanus.model.shape.dask import gaussian
from africanus.util.requirements import requires_optional
import africanus.model.wsclean

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms",
                   help="Input .MS file.")
    p.add_argument("-sm", "--sky-model", default="sky-model.txt",
                   help="Name of file containing the sky model. Default is 'sky-model.txt'")
    p.add_argument("-rc", "--row-chunks", type=int, default=10000,
                   help="Number of rows of input .MS that are processed in a single chunk. "
                        "Default is 10000.")
    p.add_argument("-mc", "--model-chunks", type=int, default=10,
                   help="Number of sky model components that are processed in a single chunk. "
                        "Default is 10.")
    p.add_argument("-iuvw", "--invert-uvw", action="store_true",
                   help="Optional. Invert UVW coordinates. Useful if we want "
                        "compare our visibilities against MeqTrees.")
    p.add_argument("-sp", "--spectra", action="store_true",
                   help="Optional. Model sources as non-flat spectra. The spectral "
                        "coefficients and reference frequency must be present in the sky model.")
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

    corrs = pol.NUM_CORR.values
    corr_types = pol.CORR_TYPE.values

    if corrs == 4:
        return [[corr_types[0], corr_types[1]],
                [corr_types[2], corr_types[3]]]  # (2, 2) shape
    elif corrs == 2:
        return [corr_types[0], corr_types[1]]    # (2, ) shape
    elif corrs == 1:
        return [corr_types[0]]                   # (1, ) shape
    else:
        raise ValueError("corrs %d not in (1, 2, 4)" % corrs)


def einsum_schema(pol,dospec):
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
        if dospec: return "srf, sfij -> srfij"
        else: return "srf, sij -> srfij"
    elif corrs in (2, 1):
        if dospec: return "srf, sfi -> srfi"
        else: return "srf, si -> srfi"
    else:
        raise ValueError("corrs %d not in (1, 2, 4)" % corrs)


def import_from_wsclean(wsclean_comp_list,dospec):
    # Read WSclean component list header
    wsclean_sources = { label: np.array(value) for label, value in africanus.model.wsclean.load(wsclean_comp_list) }
    for hh in ['Ra','Dec','I','SpectralIndex','ReferenceFrequency','LogarithmicSI','MajorAxis','MinorAxis','Orientation']:
        if hh not in wsclean_sources: raise KeyError('"{0:s}" not in header of {1:s}'.format(hh,wsclean_comp_list))
    ra=np.array(wsclean_sources['Ra'])
    dec=np.array(wsclean_sources['Dec'])
    # Load flux density at reference frequency
    fluxdens=wsclean_sources['I']
    bmaj=wsclean_sources['MajorAxis']
    bmin=wsclean_sources['MinorAxis']
    bpa=wsclean_sources['Orientation']
    if dospec:
        # Load spectral coefficients
        coeff=wsclean_sources['SpectralIndex']
        # Load reference frequency
        refrq=wsclean_sources['ReferenceFrequency']
        logsi=wsclean_sources['LogarithmicSI']
        if np.unique(logsi).shape[0]>1:
            print('Mixed log and ordinary polynomial spectral coefficients in {0:s}. Cannot deal with that. Aborting.'.format(wsclean_comp_list))
            sys.exit()
        else: logsi=np.unique(logsi)[0]
    else: coeff,refrq,logsi = None,None,None
    zero = np.zeros_like(fluxdens)
    
    return np.stack((ra,dec),axis=1),np.stack((fluxdens, zero, zero, zero),axis=1),coeff,refrq,logsi,np.stack((bmaj,bmin,bpa),axis=-1)

@requires_optional("dask.array", "xarray", "xarrayms", opt_import_error)
def predict(args):
    # Import source data from WSClean component list
    # See https://sourceforge.net/p/wsclean/wiki/ComponentList
    radec,stokes,spec_coeff,ref_freq,log_spec_ind,gaussian_shape=import_from_wsclean(args.sky_model,args.spectra)

    # OR set source data manually
    #radec = np.pi/180*np.array([
    #    [54.4, -35.8],
    #    [54.7, -35.7],
    #    [55.0, -35.6],
    #   ])
    #stokes = np.array([
    #    [10.0, 0, 0, 0],
    #    [20.0, 0, 0, 0],
    #    [30.0, 0, 0, 0],
    #   ])
    #spec_coeff = np.array([
    #    [0,],
    #    [0,],
    #    [0,],
    #   ])
    #spec_coeff = None
    #ref_freq = np.ones(radec.shape[0])*1.4e+9
    #ref_freq = None
    #log_spec_ind = False

    radec = da.from_array(radec, chunks=(args.model_chunks, 2))
    stokes = da.from_array(stokes, chunks=(args.model_chunks, 4))
    if (gaussian_shape[:,:2]!=0).sum(): # testing only on bmaj,bmin
        gaussian_components=True
        gaussian_shape = da.from_array(gaussian_shape, chunks=(args.model_chunks, 3))
    else: gaussian_components=False

    if args.spectra:
        spec_coeff = da.from_array(spec_coeff, chunks=(args.model_chunks, spec_coeff.shape[1]))
        ref_freq = da.from_array(ref_freq, chunks=(args.model_chunks,))

    # Get the support tables
    tables = support_tables(args, ["FIELD", "DATA_DESCRIPTION",
                                   "SPECTRAL_WINDOW", "POLARIZATION"])

    field_ds = tables["FIELD"]
    ddid_ds = tables["DATA_DESCRIPTION"]
    spw_ds = tables["SPECTRAL_WINDOW"]
    pol_ds = tables["POLARIZATION"]

    # List of write operations
    writes = []

    # Construct a graph for each DATA_DESC_ID
    for xds in xds_from_ms(args.ms,
                           columns=["UVW", "ANTENNA1", "ANTENNA2", "TIME"],
                           group_cols=["FIELD_ID", "DATA_DESC_ID"],
                           chunks={"row": args.row_chunks}):

        # Extract frequencies from the spectral window associated
        # with this data descriptor id
        field = field_ds[xds.attrs['FIELD_ID']]
        ddid = ddid_ds[xds.attrs['DATA_DESC_ID']]
        spw = spw_ds[ddid.SPECTRAL_WINDOW_ID.values]
        pol = pol_ds[ddid.POLARIZATION_ID.values]
        frequency = spw.CHAN_FREQ.data

        corrs = pol.NUM_CORR.values

        lm = radec_to_lm(radec, field.PHASE_DIR.data)
        uvw = -xds.UVW.data if args.invert_uvw else xds.UVW.data

        if args.spectra:
            # flux density at reference frequency ...
            # ... for logarithmic polynomial functions
            if log_spec_ind: Is=da.log(stokes[:,0,None])*frequency[None,:]**0
            # ... or for ordinary polynomial functions
            else: Is=stokes[:,0,None]*frequency[None,:]**0
            # additional terms of SED ...
            for jj in range(spec_coeff.shape[1]):
                # ... for logarithmic polynomial functions
                if log_spec_ind: Is+=spec_coeff[:,jj,None]*da.log((frequency[None,:]/ref_freq[:,None])**(jj+1))
                # ... or for ordinary polynomial functions
                else: Is+=spec_coeff[:,jj,None]*(frequency[None,:]/ref_freq[:,None]-1)**(jj+1)
            if log_spec_ind: Is=da.exp(Is)
            Qs=da.zeros_like(Is)
            Us=da.zeros_like(Is)
            Vs=da.zeros_like(Is)
            spectrum=da.stack([Is,Qs,Us,Vs],axis=-1) # stack along new axis and make it the last axis of the new array
            spectrum=spectrum.rechunk(spectrum.chunks[:2] + (spectrum.shape[2],))

        print('-------------------------------------------')
        print('Nr sources        = {0:d}'.format(stokes.shape[0]))
        print('-------------------------------------------')
        print('stokes.shape      = {0:}'.format(stokes.shape))
        print('frequency.shape   = {0:}'.format(frequency.shape))
        if args.spectra: print('Is.shape          = {0:}'.format(Is.shape))
        if args.spectra: print('spectrum.shape    = {0:}'.format(spectrum.shape))

        # (source, row, frequency)
        phase = phase_delay(lm, uvw, frequency)
        if gaussian_components: phase *= gaussian(uvw, frequency, gaussian_shape)
        # (source, frequency, corr_products)
        brightness = convert(spectrum if args.spectra else stokes, ["I", "Q", "U", "V"],
                             corr_schema(pol))

        print('brightness.shape  = {0:}'.format(brightness.shape))
        print('phase.shape       = {0:}'.format(phase.shape))
        print('-------------------------------------------')
        print('Attempting phase-brightness einsum with "{0:s}"'.format(einsum_schema(pol,args.spectra)))

        # (source, row, frequency, corr_products)
        jones = da.einsum(einsum_schema(pol,args.spectra), phase, brightness)
        print('jones.shape       = {0:}'.format(jones.shape))

        # Identify time indices
        _, time_index = da.unique(xds.TIME.data, return_inverse=True)

        # Predict visibilities
        vis = predict_vis(time_index, xds.ANTENNA1.data, xds.ANTENNA2.data,
                          None, jones, None, None, None, None)

        # Reshape (2, 2) correlation to shape (4,)
        if corrs == 4:
            vis = vis.reshape(vis.shape[:2] + (4,))

        # Assign visibilities to MODEL_DATA array on the dataset
        model_data = xr.DataArray(vis, dims=["row", "chan", "corr"])
        xds = xds.assign(MODEL_DATA=model_data)
        # Create a write to the table
        write = xds_to_table(xds, args.ms, ['MODEL_DATA'])
        # Add to the list of writes
        writes.append(write)

    # Submit all graph computations in parallel
    with ProgressBar():
        dask.compute(writes)
