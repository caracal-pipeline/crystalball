from __future__ import annotations
from typing import Literal

import numpy as np
import psutil
from dask.distributed import Client
from loguru import logger as log


def get_budget(
        nr_sources: int,
        nr_rows: int,
        nr_chans: int,
        nr_corrs: int,
        data_type: Literal["complex", "dcomplex"] = "complex", 
        num_workers: int | None = None,
        model_chunks: int = 0,
        row_chunks: int = 0,
        memory_fraction: float = 0.1,
        fudge_factor=1.25, 
        row2source_ratio=100,
        client: Client | None = None,
) -> tuple[int, int]:
    """Compute the appropriate chunk sizes for the model and data.

    Parameters
    ----------
    nr_sources : int
        Number of sources in the model.
    nr_rows : int
        Number of rows in the visibilities.
    nr_chans : int
        Number of channels in the visibilities.
    nr_corrs : int
        Number of polarizations in the visibilities.
    data_type : Literal["complex", "dcomplex"], optional
        Data type, by default "complex"
    num_workers : int | None, optional
        Number of workers, by default None
    model_chunks : int, optional
        Number of model chunks, by default 0
    row_chunks : int, optional
        Number of row chunks, by default 0
    memory_fraction : float, optional
        Fraction of memory to use, by default 0.1
    fudge_factor : float, optional
        Padding factor for memory, by default 1.25
    row2source_ratio : int, optional
        Rows per source in chunk, by default 100
    client : Client | None, optional
        Dask client in use, by default None

    Returns
    -------
    tuple[int, int]
        rows_per_chunk, sources_per_chunk

    Raises
    ------
    ValueError
        If the memory fraction is not in the interval (0,1]
    ValueError
        If only one of the row or source chunk is set
    ValueError
        If the scheduler address cannot be found
    """    
    log.info("-" * 50)
    log.info("Budgeting")
    log.info("-" * 50)

    if client is not None:
        info = client.scheduler_info()
        addr = info.get("address")
        if addr is None:
            msg = f"Cannot get the scheduler address {client=}"
            raise ValueError(msg)

        workers = info.get("workers", {})
        nworkers = len(workers)
        total_threads = sum(w["nthreads"] for w in workers.values())
        total_mem = sum([w["memory_limit"] for w in workers.values()])
        systmem = total_mem / total_threads
        nrthreads = total_threads // nworkers

        log.info("Total RAM = {0:.2f} GB", total_mem / 1024**3)
        log.info("RAM / thread = {0:.2f} GB", systmem / 1024**3)
        log.info("nr of dask workers = {0:d}", nworkers)
        log.info("nr of threads / worker = {0:d}", nrthreads)

    else:
        systmem = float(psutil.virtual_memory()[0])
        if not num_workers:
            nrthreads = psutil.cpu_count()
        else:
            nrthreads = num_workers
        log.info('system RAM = {0:.2f} GB', systmem / 1024**3)
        log.info('nr of logical CPUs = {0:d}', nrthreads)

    log.info('nr sources = {0:d}', nr_sources)
    log.info('nr rows    = {0:d}', nr_rows)
    log.info('nr chans   = {0:d}', nr_chans)
    log.info('nr corrs   = {0:d}', nr_corrs)

    data_type_str = {'complex': 'complex64', 'dcomplex': 'complex128'}[data_type]
    data_bytes = np.dtype(data_type_str).itemsize
    bytes_per_row = nr_chans * nr_corrs * data_bytes
    memory_per_row = bytes_per_row * fudge_factor

    if model_chunks and row_chunks:
        rows_per_chunk = row_chunks
        sources_per_chunk = model_chunks
        strat_type = "(user settings)"
    elif not model_chunks and not row_chunks:

        if memory_fraction > 1 or memory_fraction <= 0:
            raise ValueError('The memory fraction must be a number in the '
                             'interval (0,1]. You have set it to {0:f} '
                             '.'.format(memory_fraction))

        allowed_rows_per_thread = (systmem * memory_fraction /
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