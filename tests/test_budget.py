from contextlib import contextmanager
import logging
import psutil
from time import perf_counter

from africanus.rime import wsclean_predict

from crystalball.budget import budget

from loguru import logger
import numpy as np
import pytest


log = logging.getLogger(__file__)

@contextmanager
def catchtime() -> float:
    start = perf_counter()
    yield lambda: perf_counter() - start



@pytest.fixture
def caplog(caplog):
    handler_id = logger.add(caplog.handler, format="{message}")
    yield caplog
    logger.remove(handler_id)



def test_desktop_budget(caplog):
    sysmem = 16*(1024**3)

    # Many rows and some sources for 16 channels/4 correlations
    assert (39063, 100) == budget(np.complex64, 16, 4, sysmem, 8)

    assert (153, 100) == budget(np.complex64, 4096, 4, sysmem, 8)
    assert (153, 100) == budget(np.complex64, 4096, 4, sysmem, 4)
    assert (153, 100) == budget(np.complex64, 4096, 4, sysmem, 1)

    # It's still possible to handle many channels
    assert (1, 4) == budget(np.complex64, 4096*4096, 4, sysmem, 8)
    assert len(caplog.records) == 0

    # Exceed the system memory limit
    with pytest.raises(ValueError) as e:
        assert (1, 1) == budget(np.complex64, 4096*4096*8, 4, sysmem, 8)

    assert "impossible" in e.value.args[0]
    assert "4096MB" in e.value.args[0]      # Chunk size
    assert "16GB" in e.value.args[0]        # System memory size

    # We can fit the predict into 16MB RAM and 8 cores
    # but there's not a lot of work for each core to do
    assert (5, 100) == budget(np.complex64, 4096, 4, 8*(1024**2), 8)
    assert len(caplog.records) == 1
    assert "reduce(mul, (100, 5, 4096, 4), 1) == 8192000" in caplog.text
    assert "may not fully utilise each CPU core" in caplog.text
    assert "needs to use approximately 19MB per core" in caplog.text


@pytest.mark.parametrize("nchan", [128, 1024*1024, 2*1024*1024])
@pytest.mark.parametrize("ncorr", [4, 2])
@pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128])
def test_wsclean_budget(nchan, ncorr, complex_dtype):
    sysmem = float(psutil.virtual_memory()[0])
    processors = psutil.cpu_count()
    nrows, nsrc = budget(complex_dtype, nchan, ncorr, sysmem, processors)

    flux = np.random.random(nsrc)
    lm = np.random.random((nsrc, 2))
    spi = np.random.random((nsrc, 2))
    log_poly = np.full_like(flux, False, dtype=np.bool)
    frequency = np.linspace(.856e9, 2*.856e9, nchan)
    ref_freq = np.full_like(flux, (.856e9 + 2*.856e9)/2)
    uvw = np.random.random((nrows, 3))*10_000
    source_type = np.array(["POINT"]*nsrc)
    gauss_shape = np.zeros((nsrc, 3))

    with catchtime() as t:
        vis = wsclean_predict(uvw, lm, source_type, flux, spi, log_poly, ref_freq, gauss_shape, frequency)

    print(f"Execution time: {t():.4f} secs")
    print(f"Visibility {(nsrc,) + vis.shape} number of bytes: {vis.nbytes*ncorr/1024.**2}MB")
