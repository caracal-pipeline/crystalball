import psutil
import numpy as np

def get_budget(nr_sources,nr_rows,cb_args):
    systmem=np.float(psutil.virtual_memory()[0])
    print('-------------------------------------------')
    print('system RAM = {0:.1f} GB'.format(systmem/1024**3))
    print('nr of logical CPUs = {0:d}'.format(psutil.cpu_count()))
    print('nr sources = {0:d}'.format(nr_sources))
    print('nr rows = {0:d}'.format(nr_rows))
    print('sources per chunk = {0:d}'.format(cb_args.model_chunks))
    print('rows per chunk = {0:d}'.format(cb_args.row_chunks))
    nr_chunks = np.float(nr_sources*nr_rows)/cb_args.model_chunks/cb_args.row_chunks
    print('total nr of chunks = {0:.1f}'.format(nr_chunks))
    return 0 # Eventually this will return some relevant quantities
