from __future__ import annotations

import numpy as np
import psutil
from dask.distributed import Client
from loguru import logger as log


def get_budget(
        nr_sources, 
        nr_rows, 
        nr_chans, 
        nr_corrs, 
        data_type, 
        num_workers: int | None = None,
        model_chunks: int = 0,
        row_chunks: int = 0,
        memory_fraction: float = 0.1,
        fudge_factor=1.25, 
        row2source_ratio=100
):
    systmem = float(psutil.virtual_memory()[0])
    if not num_workers:
        nrthreads = psutil.cpu_count()
    else:
        nrthreads = num_workers

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


def get_budget_from_client(
    nr_sources,
    nr_rows,
    nr_chans,
    nr_corrs,
    data_type,
    client: Client,
    model_chunks: int = 0,
    row_chunks: int = 0,
    memory_fraction: float = 0.1,
    fudge_factor=1.25,
    row2source_ratio=100,
):

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

    log.info("-" * 50)
    log.info("Budgeting")
    log.info("-" * 50)
    log.info("Total RAM = {0:.2f} GB", total_mem / 1024**3)
    log.info("RAM / thread = {0:.2f} GB", systmem / 1024**3)
    log.info("nr of dask workers = {0:d}", nworkers)
    log.info("nr of threads / worker = {0:d}", nrthreads)
    log.info("nr sources = {0:d}", nr_sources)
    log.info("nr rows    = {0:d}", nr_rows)
    log.info("nr chans   = {0:d}", nr_chans)
    log.info("nr corrs   = {0:d}", nr_corrs)

    data_type = {"complex": "complex64", "dcomplex": "complex128"}[data_type]
    data_bytes = np.dtype(data_type).itemsize
    bytes_per_row = nr_chans * nr_corrs * data_bytes
    memory_per_row = bytes_per_row * fudge_factor

    if model_chunks and row_chunks:
        rows_per_chunk = row_chunks
        sources_per_chunk = model_chunks
        strat_type = "(user settings)"
    elif not model_chunks and not row_chunks:
        if memory_fraction > 1 or memory_fraction <= 0:
            raise ValueError(
                "The memory fraction must be a number in the "
                "interval (0,1]. You have set it to {0:f} "
                ".".format(memory_fraction)
            )

        allowed_rows_per_thread = (
            systmem * memory_fraction / (memory_per_row * nrthreads)
        )
        rows_per_chunk = int(min(nr_rows, allowed_rows_per_thread))
        sources_per_chunk = int(min(nr_sources, rows_per_chunk / row2source_ratio))
        strat_type = "(auto settings)"
    else:
        raise ValueError(
            "For now you must set both row and source chunk, or "
            "leave both unset (=0); "
            "you cannot set only one of them."
        )

    log.info("sources per chunk = {0:.0f} {1}", sources_per_chunk, strat_type)
    log.info("rows per chunk    = {0:.0f} {1}", rows_per_chunk, strat_type)

    memory_usage = rows_per_chunk * memory_per_row * nrthreads + sources_per_chunk
    log.info("expected memory usage = {0:.2f} GB", memory_usage / 1024**3)

    return rows_per_chunk, sources_per_chunk
