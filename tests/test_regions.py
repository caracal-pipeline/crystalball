from astropy.coordinates import SkyCoord, Angle
import astropy.units as units

from crystalball.region import load_regions


def test_regions(ds9_region_file):
    regions = load_regions(ds9_region_file)

    c1 = SkyCoord(299.8452923, 40.7403572, frame='fk5', unit='deg')
    assert regions[0].center.position_angle(c1) == Angle(0, unit='rad')
    assert regions[0].radius == units.Quantity(3.751, unit='arcsec')

    c2 = SkyCoord(299.8478137, 40.74316449, frame='fk5', unit='deg')
    assert regions[1].center.position_angle(c2) == Angle(0, unit='rad')
    assert regions[1].radius == units.Quantity(1, unit='arcsec')

    c3 = SkyCoord(299.846081, 40.7386852, frame='fk5', unit='deg')
    assert regions[2].center.position_angle(c3) == Angle(0, unit='rad')
    assert regions[2].radius == units.Quantity(1.915, unit='arcsec')
