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

def budget(complex_dtype, nsrc, nrow, nchan, ncorr, system_memory, processors):
    """
    A visibility predict generally computes `O(row x channel x correlation)`
    space output, computing `O(source x row x channel x correlation)` values.

    Given:

        1. Total Number of Sources
        2. Total Number of Rows
        1. Total Number of channels
        2. Total Number of correlations
        3. System Memory
        4. Number of Processors
        5. Visibility data type

    complex_dtype : np.dtype
        Complex Data type
    nsrc : int
        Number of sources
    nrow : int
        Number of rows
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
    nrow_chunks : int
        Number of rows per chunk
    nsrc_chunks : int
        Number of sources per chunk
    """
    complex_dtype = np.complex128  # wsclean_predict produces complex128 by default

    nsource_nrows = DESIRED_ELEMENTS / (nchan * ncorr)
    nrow_chunks = math.ceil(nsource_nrows / min(ROW_TO_SOURCE_RATIO, nsrc))
    nsrc_chunks = math.ceil(nsource_nrows / min(nrow_chunks, nrow))
    nrow_chunks = min(nrow_chunks, nrow)
    nsrc_chunks = min(nsrc_chunks, nsrc)

    vis_shape = (nrow_chunks, nchan, ncorr)
    vis_bytes = reduce(mul, vis_shape, np.dtype(complex_dtype).itemsize)

    while vis_bytes*processors > system_memory:
        # We can't subdivide further, complain
        if nrow_chunks == 1:
            raise ValueError(f"It's impossible to fit {processors} complex visibility "
                             f"chunks of size {vis_bytes / (1024.**2):.0f}MB and "
                             f"shape {vis_shape} across {processors} processors "
                             f"and total system memory of "
                             f"{system_memory / (1024.**3):.0f}GB")

        nrow_chunks = int(math.ceil(nrow_chunks / 2))
        vis_shape = (nrow_chunks, nchan, ncorr)
        vis_bytes = reduce(mul, vis_shape, np.dtype(complex_dtype).itemsize)

    compute_elements = nsrc_chunks*nrow_chunks*nchan*ncorr

    if compute_elements < DESIRED_ELEMENTS:
        needed_bytes = DESIRED_ELEMENTS * np.dtype(complex_dtype).itemsize / ROW_TO_SOURCE_RATIO
        log.warning("Number of visibility elements {0} == {1} "
                    "per chunk is less than the desired number of elements {2}. "
                    "crystalball may not fully utilise each CPU core. "
                    "This can occur when (1) the problem size is small or, "
                    "(2) the number of CPU cores {3} multiplied by the approximately "
                    "{4:.0f}MB crystalball needs to effectively use each core is greater "
                    "than system memory {5:.1f}GB.",
                    " x ".join(map(str, (nsrc_chunks, nrow_chunks, nchan, ncorr))),
                    compute_elements, DESIRED_ELEMENTS,
                    processors, needed_bytes / (1024.**2), system_memory / (1024.**3))

    return nrow_chunks, nsrc_chunks
