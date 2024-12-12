from contextlib import ExitStack
from hashlib import sha256
import logging
from pathlib import Path
import urllib
import urllib.request
import tarfile

from appdirs import user_cache_dir
import pytest

log = logging.getLogger(__file__)

_TEST_REGION_FILE = """
# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
fk5
circle(299.8452923,40.7403572,3.751")
circle(299.8478137,40.74316449,1.000")
circle(299.846081,40.7386852,1.915")
"""  # noqa


@pytest.fixture
def ds9_region_file(tmpdir):
    filename = str(tmpdir / "region_file.txt")

    with open(filename, "w") as f:
        f.write(_TEST_REGION_FILE)

    return filename

TART_MS = "tart.ms"
TART_MS_TAR = f"{TART_MS}.tar.xz"
TART_MS_TAR_HASH = "c9ac300b6396564e80a46e8fa302994e9b62755c21c3b532a73613680a650f24"
TART_MS_URL = f"https://ratt-public-data.s3.af-south-1.amazonaws.com/test-data/crystalball/{TART_MS_TAR}"

SKY_MODEL = "sky_model.txt"
SKY_MODEL_HASH = "c9e58c476fc3f7c588e70a7dd3ba5913aed010758a3db93b21b3310a1d1c802e"
SKY_MODEL_URL = f"https://ratt-public-data.s3.af-south-1.amazonaws.com/test-data/crystalball/{SKY_MODEL}"


def download(filename: Path, url: str, sha256hash: str):
    if filename.exists():
        with open(filename, "rb") as f:
            digest = sha256()

            while data := f.read(2**20):
                digest.update(data)

            if digest.hexdigest() == sha256hash:
                return

            filename.unlink(missing_ok=True)

    for attempt in range(3):
        try:
            with ExitStack() as stack:
                response = stack.enter_context(urllib.request.urlopen(url))
                f = stack.enter_context(open(filename, "wb"))
                digest = sha256()
                while chunk := response.read(2**20):
                    digest.update(chunk)
                    f.write(chunk)

                if digest.hexdigest() == sha256hash:
                    return

            filename.unlink(missing_ok=True)
        except urllib.error.URLError as e:
            log.error("Download of %s failed on attempt %d", url, attempt, exc_info=True)

    raise ValueError(f"Download of {url} failed {attempt} times")



@pytest.fixture(scope="session")
def tart_ms_tarfile(tmp_path_factory):
    cache_dir = Path(user_cache_dir("crystalball")) / "test-data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tart_ms_tar = cache_dir / TART_MS_TAR

    download(tart_ms_tar, TART_MS_URL, TART_MS_TAR_HASH)

    msdir = tmp_path_factory.mktemp("tartms")

    with tarfile.open(tart_ms_tar) as tar:
        tar.extractall(msdir, filter="tar")

    yield str(msdir / TART_MS)

@pytest.fixture(scope="session")
def sky_model(tmp_path_factory):
    cache_dir = Path(user_cache_dir("crystalball")) / "test-data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sky_model = cache_dir / SKY_MODEL

    download(sky_model, SKY_MODEL_URL, SKY_MODEL_HASH)

    yield str(sky_model)
