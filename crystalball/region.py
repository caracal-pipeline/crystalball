import logging
import tempfile

from regions import Regions

log = logging.getLogger(__name__)


def load_regions(region_file):
    """
    Parameters
    ----------
    region_file : str
        Region File

    Returns
    -------
    list of CircleSkyRegions
    """

    # kludge because regions cries over "FK5", wants lowercase
    with tempfile.NamedTemporaryFile() as tmpfile, open(region_file) as rf:
        tmpfile.write(rf.read().lower().encode())
        tmpfile.flush()
        include_regions = Regions.read(tmpfile.name, format='ds9')
        log.info("Read %s inclusion region(s) from %s",
                 len(include_regions), region_file)

        return include_regions
