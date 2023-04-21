# -*- coding: utf-8 -*-

from collections import namedtuple
import logging
import re

from africanus.model.wsclean.file_model import load
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS, WCSSUB_CELESTIAL
import numpy as np


log = logging.getLogger(__name__)

WSCleanModel = namedtuple("WSCleanModel", ["source_type", "radec",
                                           "flux", "spi", "ref_freq",
                                           "log_poly", "gauss_shape"])


def import_from_wsclean(wsclean_comp_list,
                        include_regions=None,
                        point_only=False,
                        num=None,
                        clean_mask_file=None,
                        percent_flux=1.0):
    """
    Imports sources from wsclean, sorted from brightest to faintest.
    If ``include_regions`` is specified only those sources within
    are returned. Similarly if ``num`` is specified, ``num`` brightest
    sources are returned.

    Parameters
    ----------
    wsclean_comp_list : str
        wsclean component list file.
    include_regions : List[:class:`regions.CircleSkyRegion`], optional
        List of valid region's. Only sources within these regions
        will be loaded.
        Defaults to ``None`` in which case, all sources are loaded.
    point_only : bool, optional
        Only include point sources. Defaults to False
    num :  integer, optional
        Number of sources to include.
        If ``None`` all sources are returned.
    clean_mask_file : str, optional
        Clean mask file.
    percent_flux : float
        Percent of flux to include with each source

    Returns
    -------
    source_type : :class:`numpy.ndarray`
        Source types of shape :code:`(source,)`
    radec : :class:`numpy.ndarray`
        Source radec coordinates of shape :code:`(source, 2)`.
    stokes : :class:`numpy.ndarray`
        Source stokes paramters of shape :code:`(source, 4)`.
    spectral_index : :class:`numpy.ndarray`
        Spectral index of shape :code:`(source, spi)`
    ref_freq : :class:`numpy.ndarray`
        Reference frequency of shape :code:`(source,)`
    log_si : bool
        Boolean indicated whether to use the logarithmic spectral index.
    gauss_shape : :class:`numpy.ndarray`
        Gaussian shape parameters of shape :code:`(source, 3)`.
        The three components are MajorAxis, MinorAxis and Orientation,
        respectively.
    """

    wsclean_comps = {column: np.asarray(values)
                     for column, values
                     in load(wsclean_comp_list)}

    if np.unique(wsclean_comps['LogarithmicSI']).shape[0] > 1:
        raise ValueError("Can't handle Mixed log and ordinary polynomial "
                         "coefficients in '%s'" % wsclean_comp_list)

    # Sort components by descending flux
    sort_index = np.ascontiguousarray(np.argsort(wsclean_comps['I'])[::-1])

    # re-sort all entries using this
    wsclean_comps = {key: value[sort_index] for key, value
                     in wsclean_comps.items()}

    log.info('%s contains %d components',
             wsclean_comp_list, len(wsclean_comps['Type']))

    # make mask of sources to include
    include = np.ones_like(wsclean_comps['Type'], bool)

    if include_regions:
        include[:] = False
        # NB: regions is *supposed* to have a sensible "contains" interface,
        # but it doesn't work as of Mar 2019. So hacking
        # a kludge for circular regions for now
        from regions import CircleSkyRegion

        if not all([type(reg) is CircleSkyRegion for reg in include_regions]):
            raise ValueError('Only circular DS( regions supported for now')

        coord = SkyCoord(wsclean_comps['Ra'], wsclean_comps['Dec'],
                         unit="rad", frame=include_regions[0].center.frame)
        include = coord.separation(
            include_regions[0].center) <= include_regions[0].radius

        for reg in include_regions[1:]:
            include |= coord.separation(reg.center) <= reg.radius

        log.info("%d of which fall within the %d inclusive regions",
                 include.sum(), len(include_regions))

    # select points
    if point_only:
        include &= (wsclean_comps['Type'] == 'POINT')
        log.info("%d of which are point sources", include.sum())

    # apply filters to component list
    wsclean_comps = {key: value[include] for key, value
                     in wsclean_comps.items()}

    # select limited number if asked
    if num is not None:
        wsclean_comps = {key: value[:num] for key, value
                         in wsclean_comps.items()}
        log.info("Selecting up %d brightest sources", num)

    if clean_mask_file:
        with fits.open(clean_mask_file) as cmf:
            header = cmf[0].header
            clean_mask = cmf[0].data

            try:
                origin = header["ORIGIN"].strip()
            except KeyError:
                raise ValueError(f"{clean_mask_file} FITS header "
                                 f"doesn't contain an ORIGIN key")
            else:
                if m := re.match("^SoFiA (?P<version>\d+\.\d+\.\d+)$", origin):
                    major, _, _ = map(int, m.group("version").split("."))
                    if major < 2:
                        raise ValueError(f"SoFiA major version is less than 2: {origin}")
                else:
                    raise ValueError(f"{clean_mask_file} ORIGIN '{origin}' "
                                     f"doesn't contain a valid SoFia version")

            if clean_mask.ndim > 2:
                clean_mask = clean_mask[..., 0]
            elif clean_mask.ndim != 2:
                raise ValueError(f"data shape {clean_mask.shape} in "
                                 f"{clean_mask_file} does not have two axes")

            wcsin = WCS(header, naxis=[WCSSUB_CELESTIAL])

            ra = np.rad2deg(wsclean_comps["Ra"])
            dec = np.rad2deg(wsclean_comps["Dec"])
            flux = wsclean_comps["I"]
            x, y = wcsin.wcs_world2pix(ra, dec, 0)
            x, y = np.rint(x).astype(int), np.rint(y).astype(int)

            log.info("Grouping sources by id's in %s", clean_mask_file)

            source_ids = clean_mask[x, y]
            nr_sources = clean_mask.max()
            source_id_range = np.arange(1, nr_sources + 1)
            integrated_fluxes = np.array([flux[sid == source_ids].sum() for sid in source_id_range])

            flux_sort_index = np.argsort(integrated_fluxes)[::-1]
            integrated_fluxes = integrated_fluxes[flux_sort_index]
            total_flux = integrated_fluxes.sum()

            sorted_ids = source_id_range[flux_sort_index]
            field_flux_fraction = np.cumsum(integrated_fluxes) / total_flux
            brightest = sorted_ids[field_flux_fraction <= percent_flux]
            mask = np.in1d(source_ids, brightest)
            print(f"Selected {len(brightest)} brightest sources out of {nr_sources} total")
            print(f"Selected {mask.sum()} components out of {mask.size} total")
            log.info("Selecting %d brightest sources out of %d total",
                    len(brightest), len(source_ids))
            wsclean_comps = {k: v[mask] for k, v in wsclean_comps.items()}


    # print if small subset
    if (num is not None and num < 100) or include_regions:
        flux_types = zip(wsclean_comps['I'], wsclean_comps['Type'])
        it = enumerate(sorted(flux_types, reverse=True))
        for i, (flux, src_type) in it:
            log.info('%d: %s %s Jy', i, src_type, flux)

    log.info('Total flux of %d selected components is %f Jy',
             len(wsclean_comps['I']), wsclean_comps['I'].sum())

    # Create radec array
    radec = np.concatenate((wsclean_comps['Ra'][:, None],
                            wsclean_comps['Dec'][:, None]),
                           axis=1)

    # Create gaussian shapes
    gauss_shape = np.stack((wsclean_comps['MajorAxis'],
                            wsclean_comps['MinorAxis'],
                            wsclean_comps['Orientation']),
                           axis=-1)

    return WSCleanModel(wsclean_comps['Type'],
                        radec,
                        wsclean_comps['I'],
                        wsclean_comps['SpectralIndex'],
                        wsclean_comps['ReferenceFrequency'],
                        wsclean_comps['LogarithmicSI'],
                        gauss_shape)
