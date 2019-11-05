# -*- coding: utf-8 -*-

import logging

from africanus.model.wsclean.file_model import load
from astropy.coordinates import SkyCoord
import numpy as np


log = logging.getLogger(__name__)


def import_from_wsclean(wsclean_comp_list, include_regions=[],
                        point_only=False, num=None):
    """
    Imports sources from wsclean, sorted from brightest to faintest.
    If ``include_regions`` is specified only those sources within
    are returned. Similarly if ``num`` is specified, ``num`` brightest
    sources are returned.

    Parameters
    ----------
    wsclean_comp_list : str
        wsclean component list file.
    include_regions : List[:class:`regions.CircleSkyRegion`]
        List of valid region's. Only sources within these regions
        will be loaded.
        Defaults to ``[]`` in which case, all sources are loaded.
    point_only : bool, optional
        Only include point sources. Defaults to False
    num :  integer, optional
        Number of sources to include.
        If ``None`` all sources are returned.

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

    # Create stokes array
    flux = wsclean_comps['I']
    stokes = np.empty((flux.shape[0], 4), dtype=flux.dtype)
    stokes[:, 0] = flux
    stokes[:, 1:] = 0

    # Create gaussian shapes
    gauss_shape = np.stack((wsclean_comps['MajorAxis'],
                            wsclean_comps['MinorAxis'],
                            wsclean_comps['Orientation']),
                           axis=-1)

    log_si = wsclean_comps['LogarithmicSI'][0] if len(flux) > 0 else False

    return (wsclean_comps['Type'], radec, stokes,
            wsclean_comps['SpectralIndex'],
            wsclean_comps['ReferenceFrequency'],
            log_si,
            gauss_shape)
