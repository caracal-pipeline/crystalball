import pytest

_TEST_REGION_FILE = """
# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
fk5
circle(299.8452923,40.7403572,3.751")
circle(299.8478137,40.74316449,0.000")
circle(299.846081,40.7386852,1.915")
"""


@pytest.fixture
def ds9_region_file(tmpdir):
    filename = str(tmpdir / "region_file.txt")

    with open(filename, "w") as f:
        f.write(_TEST_REGION_FILE)

    return filename
