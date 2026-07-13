"""S2 — unit tests for pipeline/alan_hires.py on tiny synthetic inputs."""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from pipeline import alan_hires


CRS = "EPSG:26971"  # NAD83 / Illinois East — same as the Chicago dev pipeline


# ----- apply_calibration ----------------------------------------------------


def test_apply_calibration_linear_transform():
    dn = np.array([0, 10, 100], dtype="uint16")
    got = alan_hires.apply_calibration(dn, scale=0.1, offset=1.0, nodata=None)
    assert got.tolist() == pytest.approx([1.0, 2.0, 11.0])


def test_apply_calibration_nodata_becomes_nan():
    dn = np.array([0, 65535, 50], dtype="uint16")
    got = alan_hires.apply_calibration(dn, scale=1.0, offset=0.0, nodata=65535)
    assert got[0] == 0.0
    assert np.isnan(got[1])
    assert got[2] == 50.0


# ----- fixtures: tiny GeoTIFFs the sampler can open -------------------------


def _write_raster(
    path: Path,
    array: np.ndarray,
    origin_xy: tuple[float, float],
    pixel_size: float,
    nodata: float | None = None,
) -> None:
    """Write a single-band raster in CRS above. ``array`` is (rows, cols)."""
    height, width = array.shape
    transform = from_origin(origin_xy[0], origin_xy[1], pixel_size, pixel_size)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=height, width=width, count=1,
        dtype=array.dtype, crs=CRS, transform=transform, nodata=nodata,
    ) as dst:
        dst.write(array, 1)


@pytest.fixture()
def tiny_pan(tmp_path: Path) -> Path:
    """4x4 raster at 10 m resolution, origin (0, 40). Pixel (row, col) has
    DN = row*10 + col so we can predict what any sampled point returns."""
    arr = np.arange(16, dtype="uint16").reshape(4, 4) * 10  # 0..150
    path = tmp_path / "pan.tif"
    _write_raster(path, arr, origin_xy=(0.0, 40.0), pixel_size=10.0, nodata=None)
    return path


@pytest.fixture()
def tiny_mss(tmp_path: Path) -> Path:
    """4x4 3-band raster at 40 m. Band 1 (blue) values are distinct from bands
    2/3 so we can prove the sampler picked the right band."""
    blue = np.full((4, 4), 5, dtype="uint16")
    green = np.full((4, 4), 50, dtype="uint16")
    red = np.full((4, 4), 500, dtype="uint16")
    stack = np.stack([blue, green, red], axis=0)
    path = tmp_path / "mss.tif"
    transform = from_origin(0.0, 160.0, 40.0, 40.0)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=4, width=4, count=3,
        dtype="uint16", crs=CRS, transform=transform, nodata=0,
    ) as dst:
        dst.write(stack)
    return path


def _points_gdf(xy_pairs: list[tuple[float, float]]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"id": [f"b{i}" for i in range(len(xy_pairs))]},
        geometry=[Point(x, y) for x, y in xy_pairs],
        crs=CRS,
    )


# ----- sample_band_at_centroids ---------------------------------------------


def test_sample_pan_returns_known_dn_after_calibration(tiny_pan: Path):
    # Pixel centers: (5, 35), (15, 35), (5, 25), (15, 25). Row 0 col 0 = DN 0,
    # row 0 col 1 = DN 10, row 1 col 0 = DN 40. Calibration scale=2, offset=1.
    gdf = _points_gdf([(5.0, 35.0), (15.0, 35.0), (5.0, 25.0)])
    got = alan_hires.sample_band_at_centroids(
        tiny_pan, gdf, band_index=1, scale=2.0, offset=1.0
    )
    assert got.iloc[0] == pytest.approx(0 * 2.0 + 1.0)
    assert got.iloc[1] == pytest.approx(10 * 2.0 + 1.0)
    assert got.iloc[2] == pytest.approx(40 * 2.0 + 1.0)


def test_sample_outside_raster_footprint_is_null(tiny_pan: Path):
    # Point far outside the (0..40, 0..40) raster window.
    gdf = _points_gdf([(500.0, 500.0)])
    got = alan_hires.sample_band_at_centroids(
        tiny_pan, gdf, band_index=1, scale=1.0, offset=0.0
    )
    assert np.isnan(got.iloc[0])


def test_sample_at_nodata_pixel_is_null(tmp_path: Path):
    arr = np.array([[9999, 5], [5, 5]], dtype="uint16")
    path = tmp_path / "with_nodata.tif"
    _write_raster(path, arr, origin_xy=(0.0, 20.0), pixel_size=10.0, nodata=9999)
    gdf = _points_gdf([(5.0, 15.0), (15.0, 15.0)])  # first hits nodata pixel
    got = alan_hires.sample_band_at_centroids(
        path, gdf, band_index=1, scale=1.0, offset=0.0
    )
    assert np.isnan(got.iloc[0])
    assert got.iloc[1] == pytest.approx(5.0)


def test_sample_picks_requested_band_from_multi_band_raster(tiny_mss: Path):
    gdf = _points_gdf([(20.0, 100.0)])  # any interior point
    blue = alan_hires.sample_band_at_centroids(tiny_mss, gdf, 1, 1.0, 0.0)
    green = alan_hires.sample_band_at_centroids(tiny_mss, gdf, 2, 1.0, 0.0)
    red = alan_hires.sample_band_at_centroids(tiny_mss, gdf, 3, 1.0, 0.0)
    assert blue.iloc[0] == 5
    assert green.iloc[0] == 50
    assert red.iloc[0] == 500


def test_sample_band_index_out_of_range_raises(tiny_pan: Path):
    gdf = _points_gdf([(5.0, 35.0)])
    with pytest.raises(ValueError, match="out of range"):
        alan_hires.sample_band_at_centroids(tiny_pan, gdf, band_index=2, scale=1.0, offset=0.0)


@pytest.mark.filterwarnings("ignore:Geometry is in a geographic CRS")
def test_sample_reprojects_centroids_into_raster_crs(tiny_pan: Path):
    """A point supplied in WGS84 should still hit the raster if its projected
    location falls within bounds. We use the raster's origin as the reference:
    the projected point (5, 35) in EPSG:26971 corresponds to some WGS84 lat/lon
    — round-trip it through pyproj to confirm the sampler handles the reproj."""
    proj_gdf = _points_gdf([(5.0, 35.0)])
    wgs_gdf = proj_gdf.to_crs("EPSG:4326")
    got = alan_hires.sample_band_at_centroids(
        tiny_pan, wgs_gdf, band_index=1, scale=1.0, offset=0.0
    )
    # Same pixel as the projected-CRS test above → DN 0.
    assert got.iloc[0] == pytest.approx(0.0)


# ----- attach_sdgsat_columns -----------------------------------------------


BANDS = {
    "pan":  {"index": 1, "resolution_m": 10, "product_suffix": "PAN"},
    "blue": {"index": 1, "resolution_m": 40, "product_suffix": "MSS"},
}
RADIO = {"scale": 1.0, "offset": 0.0}


def test_attach_adds_both_columns_from_present_rasters(tiny_pan: Path, tiny_mss: Path):
    gdf = _points_gdf([(15.0, 35.0), (20.0, 100.0)])
    out = alan_hires.attach_sdgsat_columns(gdf, tiny_pan, tiny_mss, BANDS, RADIO)
    assert "alan_sdgsat" in out.columns
    assert "alan_sdgsat_blue" in out.columns
    assert out["alan_sdgsat"].notna().any()
    assert out["alan_sdgsat_blue"].notna().any()


def test_attach_missing_raster_yields_all_null_column(tiny_pan: Path):
    """No MSS mosaic on disk → alan_sdgsat_blue is all-NaN, not missing."""
    gdf = _points_gdf([(5.0, 35.0)])
    out = alan_hires.attach_sdgsat_columns(gdf, tiny_pan, None, BANDS, RADIO)
    assert "alan_sdgsat_blue" in out.columns
    assert out["alan_sdgsat_blue"].isna().all()
    # PAN column should still have real values.
    assert out["alan_sdgsat"].notna().all()


def test_attach_preserves_nulls_outside_coverage(tiny_pan: Path):
    # Two points: one inside raster bounds, one 1000 m away outside them.
    gdf = _points_gdf([(15.0, 35.0), (2000.0, 2000.0)])
    out = alan_hires.attach_sdgsat_columns(gdf, tiny_pan, None, BANDS, RADIO)
    assert out["alan_sdgsat"].notna().iloc[0]
    assert np.isnan(out["alan_sdgsat"].iloc[1])
