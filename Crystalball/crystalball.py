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
    from astropy.coordinates import Angle, SkyCoord
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
from africanus.model.wsclean.file_model import load

import casacore.tables

def create_parser():
    p = argparse.ArgumentParser()
    p.add_argument("ms",
                   help="Input .MS file.")
    p.add_argument("-sm", "--sky-model", default="sky-model.txt",
                   help="Name of file containing the sky model. Default is 'sky-model.txt'")
    p.add_argument("-o", "--output-column", default="MODEL_DATA",
                   help="Output visibility column. Default is '%(default)s'")
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
    p.add_argument("-w", "--within", type=str, 
                   help="Optional. Give JS9 region file. Only sources within those regions will be "
                        "included.")
    p.add_argument("-po", "--points-only", action="store_true",
                   help="Select only point-type sources.")
    p.add_argument("-ns", "--num-sources", type=int, default=0, metavar="N",
                   help="Select only N brightest sources.")
    p.add_argument("-j", "--num-workers", type=int, default=0, metavar="N",
                   help="Explicitly set the number of worker threads.")
                        

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


def import_from_wsclean(wsclean_comp_list, include_regions=[], point_only=False, num=None):
    wsclean_comps=load(wsclean_comp_list)
    wsclean_comps=dict(zip([jj[0] for jj in wsclean_comps], [np.array(jj[1]) for jj in wsclean_comps]))
    if np.unique(wsclean_comps['LogarithmicSI']).shape[0]>1:
        print('Mixed log and ordinary polynomial spectral coefficients in {0:s}. Cannot deal with that. Aborting.'.format(wsclean_comp_list))
        sys.exit()
        
    # create a sorting by descending flux
    sort_tuples = sorted([(flux, i) for i, flux in enumerate(wsclean_comps['I'])], reverse=True)
    sort_index = [i for flux, i in sort_tuples]
    
    # re-sort all entries using this
    wsclean_comps = { key: value[sort_index] for key, value in wsclean_comps.items() }
    
    print('{} contains {} components'.format(wsclean_comp_list, len(wsclean_comps['Type'])))
    
    # make mask of sources to include
    include = np.ones_like(wsclean_comps['Type'], bool)

    if include_regions:
        include[:] = False
        ## NB: regions is *supposed* to have a sensible "contains" interface, but it doesn't work as of Mar 2019. So hacking
        ## a kludge for circular regions for now
        from regions import CircleSkyRegion
        if not all([type(reg) is CircleSkyRegion for reg in include_regions]):
            print('Only circular DS( regions supported for now')
            sys.exit()
        coord = SkyCoord(wsclean_comps['Ra'], wsclean_comps['Dec'], unit="rad", frame=include_regions[0].center.frame)
        include = (coord.separation(include_regions[0].center) <= include_regions[0].radius)
        for reg in include_regions[1:]:
            include |= coord.separation(reg.center) <= reg.radius
        print('{} of which fall within the {} inclusion region(s)'.format(include.sum(), len(include_regions)))

    # select points
    if point_only:
        include &= (wsclean_comps['Type'] == 'POINT')
        print('{} of which are point sources'.format(include.sum()))
    
    # apply filters to component list
    wsclean_comps = { key: value[include] for key, value in wsclean_comps.items() }
    
    # select limited number if asked
    if num is not None:
        wsclean_comps = { key: value[:num] for key, value in wsclean_comps.items() }
        print('Selecting up to {} brightest sources'.format(num))
    
    # print if small subset
    if num < 100 or include_regions:
        for i, (srctype, flux) in enumerate(sorted(zip(wsclean_comps['I'], wsclean_comps['Type']), reverse=True)):
            print('{}: {} {} Jy'.format(i, srctype, flux))

    print('Total flux of {} selected components is {} Jy'.format(len(wsclean_comps['I']), wsclean_comps['I'].sum()))
    

    return wsclean_comps['Type'],\
        np.concatenate((wsclean_comps['Ra'][:,None],wsclean_comps['Dec'][:,None]),axis=1),\
        np.concatenate((wsclean_comps['I'][:,None],np.zeros((wsclean_comps['I'].shape[0],3))),axis=1),\
        wsclean_comps['SpectralIndex'],\
        wsclean_comps['ReferenceFrequency'],\
        wsclean_comps['LogarithmicSI'][0],\
        np.stack((wsclean_comps['MajorAxis'],wsclean_comps['MinorAxis'],wsclean_comps['Orientation']),axis=-1)


@requires_optional("dask.array", "xarray", "xarrayms", opt_import_error)
def predict(args):
    # get inclusion regions
    include_regions = []
    if args.within:
        from regions import read_ds9
        import tempfile
        # kludge because regions cries over "FK5", wants lowercase
        with tempfile.NamedTemporaryFile() as tmpfile, open(args.within) as regfile:
            tmpfile.write(regfile.read().lower())
            tmpfile.flush()
            include_regions = read_ds9(tmpfile.name)
            print("read {} inclusion region(s) from {}".format(len(include_regions), args.within))

    # Import source data from WSClean component list
    # See https://sourceforge.net/p/wsclean/wiki/ComponentList
    comp_type,radec,stokes,spec_coeff,ref_freq,log_spec_ind,gaussian_shape=import_from_wsclean(args.sky_model, include_regions=include_regions,
            point_only=args.points_only,
            num=args.num_sources or None)

    # check output column
    ms = casacore.tables.table(args.ms, readonly=False)
    if args.output_column not in ms.colnames():
        print('inserting new column {}'.format(args.output_column))
        desc = ms.getcoldesc("DATA")
        desc['name'] = args.output_column
        desc['comment'] = desc['comment'].replace(" ", "_")  # python version hates spaces, who knows why
        dminfo = ms.getdminfo("DATA")
        dminfo["NAME"] =  "{}-{}".format(dminfo["NAME"], args.output_column)
        ms.addcols(desc, dminfo)
    ms.close()

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
    if (comp_type=='GAUSSIAN').sum():
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
        # If at least one Gaussian component is present in the component list then all
        # sources are modelled as Gaussian components (Delta components have zero width)
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
        print('-------------------------------------------')
        if gaussian_components: print('Some Gaussian sources found')
        else: print('All sources are Delta functions')
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
        model_data = xr.DataArray(vis, dims=["row", "chan", "corr"])
        xds = xds.assign(**{args.output_column: model_data})
        # Create a write to the table
        write = xds_to_table(xds, args.ms, [args.output_column])
        # Add to the list of writes
        writes.append(write)

    # Submit all graph computations in parallel
    if args.num_workers:
        with ProgressBar(), dask.config.set(num_workers=args.num_workers):
            dask.compute(writes)
    else:
        with ProgressBar():
            dask.compute(writes)

