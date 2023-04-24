import pytest

from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import numpy as np

from crystalball.wsclean import import_from_wsclean

NAXIS1 = 120
NAXIS2 = 120
NAXIS3 = 231
RESTFRE = 1.42040575177E+09

@pytest.fixture
def clean_mask_fits_header():
    return {
        "SIMPLE":                  "T",
        "BITPIX":                   32,
        "NAXIS":                     3,
        "NAXIS1":               NAXIS1,
        "NAXIS2":               NAXIS2,
        "NAXIS3":               NAXIS3,
        "CTYPE1":           "RA---SIN",
        "CRPIX1":    6.10000000000E+01,
        "CDELT1":   -8.33333333333E-03,
        "CRVAL1":   -1.60271625000E+02,
        "CUNIT1":           "deg",
        "CTYPE2":           "DEC--SIN",
        "CRPIX2":    6.10000000000E+01,
        "CDELT2":    8.33333333333E-03,
        "CRVAL2":   -2.10391111111E+01,
        "CUNIT2":           "deg",
        "CTYPE3":           "VRAD",
        "CRPIX3":    1.00000000000E+00,
        "CDELT3":   -1.37845719375E+03,
        "CRVAL3":    8.32952108724E+05,
        "ORIGIN":        "SoFiA 2.3.1",
        "RESTFREQ":             RESTFRE,
        "EQUINOX":    2.00000000000E+03,
        "BUNIT":                     ""
    }

@pytest.fixture
def model_sources():
    # source_id, x-offset, y-offset, z-offset
    return [
        (1, -1, -1, 100),
        (1, -2, 3, 100),
        (2, 4, 5, 100),
        (2, 3, 4, -100),
        (2, 3, 2, -100),
        (3, 0, 1, -25)]


@pytest.fixture
def wsclean_model_and_clean_mask(clean_mask_fits_header, model_sources, tmp_path):
    """ Generate a synthetic wsclean model file and an associated clean mask """
    header = clean_mask_fits_header
    naxes = header["NAXIS"]
    shape = tuple((header[f"NAXIS{ax}"] for ax in range(1, naxes + 1)))
    cube = np.zeros(tuple(reversed(shape)), dtype=np.int32)

    wcs = WCS(clean_mask_fits_header)

    half_nax1 = NAXIS1 // 2
    half_nax2 = NAXIS2 // 2
    half_nax3 = NAXIS3 // 2

    lines = [f"Format = Name, Type, Ra, Dec, I, SpectralIndex, "
             f"LogarithmicSI, ReferenceFrequency='{RESTFRE}', "
             f"MajorAxis, MinorAxis, Orientation"]

    for source_id, xoff, yoff, zoff in model_sources:
        assert source_id > 0, f"{source_id} must be > 0"
        nx = half_nax1 + xoff
        ny = half_nax2 + yoff
        nz = half_nax3 + zoff
        cube[nz, ny, nx] = source_id

        world = wcs.pixel_to_world(nx, ny, nz)
        ra = world[0].ra.to_string(sep=":", unit=u.hour)
        dec = world[0].dec.to_string(sep=".", unit=u.deg)
        freq = world[1].to(u.hertz).value
        lines.append(f"s0c{source_id},POINT,{ra},{dec},1.0,[0.01,0.01],false,{freq},,,")

    wsclean_filename = tmp_path / "model.txt"

    with open(wsclean_filename, "w") as f:
        f.write("\n".join(lines))

    clean_filename = tmp_path / "clean_mask.fits"

    primary_hdu = fits.PrimaryHDU(cube, header=fits.Header(header))
    primary_hdu.writeto(str(clean_filename), overwrite=True)

    return str(wsclean_filename), str(clean_filename)


def test_clean_mask(wsclean_model_and_clean_mask):
    # 3 sources
    # 1: 2 components of 1.0 flux
    # 2: 3 component of 1.0 flux
    # 3: 1 component of 1.0 flux

    model_file, clean_file = wsclean_model_and_clean_mask

    results = import_from_wsclean(model_file, clean_mask_file=clean_file, percent_flux=1.0)
    assert len(results.source_type) == 6

    results = import_from_wsclean(model_file, clean_mask_file=clean_file, percent_flux=0.67)
    assert len(results.source_type) == 4

    results = import_from_wsclean(model_file, clean_mask_file=clean_file, percent_flux=0.66)
    assert len(results.source_type) == 3

    results = import_from_wsclean(model_file, clean_mask_file=clean_file, percent_flux=0.17)
    assert len(results.source_type) == 1

    # TODO(sjperkins): Perhaps select some components if it's not possible to satisfy
    results = import_from_wsclean(model_file, clean_mask_file=clean_file, percent_flux=0.16)
    assert len(results.source_type) == 0