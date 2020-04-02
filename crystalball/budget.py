import psutil
import numpy as np


def get_budget(nr_sources, nr_rows, nr_chans, nr_corrs, data_type, cb_args,
               fudge_factor=1.25, row2source_ratio=100):
    systmem = np.float(psutil.virtual_memory()[0])
    if not cb_args.num_workers:
        nrthreads = psutil.cpu_count()
    else:
        nrthreads = cb_args.num_workers

    print('-------------------------------------------')
    print('system RAM = {0:.2f} GB'.format(systmem/1024**3))
    print('nr of logical CPUs = {0:d}'.format(nrthreads))
    print('nr sources = {0:d}'.format(nr_sources))
    print('nr rows    = {0:d}'.format(nr_rows))
    print('nr chans   = {0:d}'.format(nr_chans))
    print('nr corrs   = {0:d}'.format(nr_corrs))

    data_type = {'complex': 'complex64', 'dcomplex': 'complex128'}[data_type]
    data_bytes = np.dtype(data_type).itemsize
    bytes_per_row = nr_chans*nr_corrs*data_bytes
    memory_per_row = bytes_per_row * fudge_factor

    if cb_args.model_chunks and cb_args.row_chunks:
        rows_per_chunk = cb_args.row_chunks
        sources_per_chunk = cb_args.model_chunks
        strat_type = "(user settings)"
    elif not cb_args.model_chunks and not cb_args.row_chunks:
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

    print('sources per chunk = {0:.0f} {1}'
          .format(sources_per_chunk, strat_type))
    print('rows per chunk    = {0:.0f} {1}'
          .format(rows_per_chunk, strat_type))

    memory_usage = (rows_per_chunk * memory_per_row * nrthreads
                    + sources_per_chunk)
    print('expected memory usage = {0:.2f} GB'.format(memory_usage/1024**3))

    return rows_per_chunk, sources_per_chunk
