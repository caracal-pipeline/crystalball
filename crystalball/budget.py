from functools import reduce
from operator import mul

import math

from loguru import logger as log
import psutil
import numpy as np


def get_budget(nr_sources, nr_rows, nr_chans, nr_corrs, data_type, cb_args,
               fudge_factor=1.25, row2source_ratio=100):
    systmem = np.float(psutil.virtual_memory()[0])
    if not cb_args.num_workers:
        nrthreads = psutil.cpu_count()
    else:
        nrthreads = cb_args.num_workers

    log.info('-' * 50)
    log.info('Budgeting')
    log.info('-' * 50)
    log.info('system RAM = {0:.2f} GB', systmem / 1024**3)
    log.info('nr of logical CPUs = {0:d}', nrthreads)
    log.info('nr sources = {0:d}', nr_sources)
    log.info('nr rows    = {0:d}', nr_rows)
    log.info('nr chans   = {0:d}', nr_chans)
    log.info('nr corrs   = {0:d}', nr_corrs)

    data_type = {'complex': 'complex64', 'dcomplex': 'complex128'}[data_type]
    data_bytes = np.dtype(data_type).itemsize
    bytes_per_row = nr_chans * nr_corrs * data_bytes
    memory_per_row = bytes_per_row * fudge_factor

    if cb_args.model_chunks and cb_args.row_chunks:
        rows_per_chunk = cb_args.row_chunks
        sources_per_chunk = cb_args.model_chunks
        strat_type = "(user settings)"
    elif not cb_args.model_chunks and not cb_args.row_chunks:

        if cb_args.memory_fraction > 1 or cb_args.memory_fraction <= 0:
            raise ValueError('The memory fraction must be a number in the '
                             'interval (0,1]. You have set it to {0:f} '
                             '.'.format(cb_args.memory_fraction))

        allowed_rows_per_thread = (systmem * cb_args.memory_fraction /
                                   (memory_per_row * nrthreads))
        rows_per_chunk = int(min(nr_rows, allowed_rows_per_thread))
        sources_per_chunk = int(min(nr_sources,
                                    rows_per_chunk / row2source_ratio))
        strat_type = "(auto settings)"
    else:
        raise ValueError('For now you must set both row and source chunk, or '
                         'leave both unset (=0); '
                         'you cannot set only one of them.')

    log.info('sources per chunk = {0:.0f} {1}', sources_per_chunk, strat_type)
    log.info('rows per chunk    = {0:.0f} {1}', rows_per_chunk, strat_type)

    memory_usage = (rows_per_chunk * memory_per_row * nrthreads
                    + sources_per_chunk)
    log.info('expected memory usage = {0:.2f} GB', memory_usage / 1024**3)

    return rows_per_chunk, sources_per_chunk


DESIRED_ELEMENTS = 250_000_000
ROW_TO_SOURCE_RATIO = 100

def budget(complex_dtype, nchan, ncorr, system_memory, processors):
    """
    A visibility predict generally computes `O(row x channel x correlation)`
    space output, computing `O(source x row x channel x correlation)` values.

    Given:

        1. Number of channels
        2. Number of correlations
        3. System Memory
        4. Number of Processors
        5. Visibility data type

    complex_dtype : np.dtype
        Complex Data type
    nchan : int
        Number of channels
    ncorr : int
        Number of correlations
    system_memory : int
        Available system memory.
    processors : int
        Number of available processors

    Returns
    -------
    nrows : int
        Number of rows per chunk
    nsrcs : int
        Number of sources per chunk
    """
    nsource_nrows = DESIRED_ELEMENTS / (nchan * ncorr)
    nrows = math.ceil(nsource_nrows / ROW_TO_SOURCE_RATIO)
    nsrc = math.ceil(nsource_nrows / nrows)

    vis_shape = (nrows, nchan, ncorr)
    vis_bytes = reduce(mul, vis_shape, np.dtype(complex_dtype).itemsize)

    while vis_bytes*processors > system_memory:
        # We can't subdivide further, complain
        if nrows == 1:
            raise ValueError(f"It's impossible to fit {processors} complex visibility "
                             f"chunks of size {vis_bytes / (1024.**2):.0f}MB and "
                             f"shape {vis_shape} across {processors} processors "
                             f"and total system memory of "
                             f"{system_memory / (1024.**3):.0f}GB")

        nrows = int(math.ceil(nrows / 2))
        vis_shape = (nrows, nchan, ncorr)
        vis_bytes = reduce(mul, vis_shape, np.dtype(complex_dtype).itemsize)

    compute_elements = nsrc*nrows*nchan*ncorr

    if compute_elements < DESIRED_ELEMENTS:
        # Sources don't contribute to visibility memory as
        # they are accumulated into visibility buffers
        needed_bytes = DESIRED_ELEMENTS * np.dtype(complex_dtype).itemsize / nsrc
        log.warning("Number of visibility elements reduce(mul, {0}, 1) == {1}) "
                    "per chunk is less the desired number of elements {2}. "
                    "crystalball may not fully utilise each CPU core. "
                    "This can occur when system memory is small "
                    "relative to the number of CPU cores on the system. "
                    "crystalball needs to use approximately {3:.0f}MB per core",
                    (nsrc, nrows, nchan, ncorr), compute_elements,
                     DESIRED_ELEMENTS, needed_bytes / (1024.**2))

    return nrows, nsrc
