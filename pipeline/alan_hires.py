"""S2 — High-resolution nighttime radiance sampling (SDGSAT-1 Glimmer).

Attaches per-building lighting from SDGSAT-1 Glimmer as a parallel column
to VIIRS ``alan_radiance``:

    alan_sdgsat        — panchromatic (10 m, 444–910 nm)
    alan_sdgsat_blue   — blue band    (40 m, 424–526 nm)

At 500 m/pixel every building in a VIIRS pixel shares one radiance value; at
10–40 m Glimmer can tell an isolated lit tower apart from its dark neighbor.
The blue band matters because collision literature (Tan et al. 2023) implicates
short-wavelength light as disproportionately attractive to nocturnal migrants,
and VIIRS DNB under-senses blue.

Design mirrors [alan.py](alan.py):

* **Centroid sampling.** One value per building. Simpler than polygon means
  and matches the VIIRS path so buildings are directly comparable across the
  two lighting sources.
* **Null-preserving.** Buildings outside scene coverage (or on a nodata /
  cloud pixel) get NaN — they must fall back to VIIRS downstream, not silently
  read zero. Same philosophy as [risk.py](risk.py) null handling.
* **Physical radiance.** DN → radiance via the SDGSAT-1 Glimmer spec:
  ``radiance = DN * scale + offset``  (W·m⁻²·sr⁻¹·µm⁻¹). Scale/offset default
  from config; override with the per-scene metadata XML if available.

No network calls here — mosaics come from ``scripts/fetch_sdgsat.py``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


def apply_calibration(
    dn: np.ndarray,
    scale: float,
    offset: float,
    nodata: float | None,
) -> np.ndarray:
    """DN → physical radiance. Nodata values become NaN.

    ``radiance = DN * scale + offset``  per SDGSAT-1 Glimmer data spec.
    Kept as a small pure function so the conversion is unit-testable on a
    tiny synthetic array without needing a raster on disk.
    """
    out = dn.astype("float64") * float(scale) + float(offset)
    if nodata is not None:
        out = np.where(dn == nodata, np.nan, out)
    return out


def _centroids_in_raster_crs(
    gdf: gpd.GeoDataFrame, raster_crs: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Project building centroids into the raster's CRS, return (xs, ys)."""
    if gdf.crs is None:
        raise ValueError("gdf must have a CRS set")
    centroids = gdf.geometry.centroid.to_crs(raster_crs)
    return centroids.x.to_numpy(), centroids.y.to_numpy()


def sample_band_at_centroids(
    raster_path: Path,
    gdf: gpd.GeoDataFrame,
    band_index: int,
    scale: float,
    offset: float,
) -> pd.Series:
    """Sample one band of ``raster_path`` at each building centroid.

    Returns a float64 Series indexed like ``gdf``. Values are physical
    radiance (post-calibration). Buildings outside the raster footprint or
    on a nodata pixel get NaN — do NOT silently fill. The null contract
    lets the risk model fall back to VIIRS for those buildings.
    """
    if gdf.empty:
        return pd.Series([], dtype="float64", index=gdf.index)

    with rasterio.open(raster_path) as ds:
        if band_index < 1 or band_index > ds.count:
            raise ValueError(
                f"band_index {band_index} out of range for {raster_path.name} "
                f"(has {ds.count} band(s))"
            )
        xs, ys = _centroids_in_raster_crs(gdf, ds.crs)
        nodata = ds.nodata
        left, bottom, right, top = ds.bounds

        # Bounds-check first: rasterio.sample fills out-of-window points with
        # the fill value (usually 0) when no nodata is set on the band, which
        # would silently look like "dark pixel." Force NaN there instead.
        in_bounds = (xs >= left) & (xs < right) & (ys >= bottom) & (ys < top)

        sampled = np.full(len(xs), np.nan, dtype="float64")
        if in_bounds.any():
            coords = list(zip(xs[in_bounds], ys[in_bounds]))
            # ds.sample with indexes=[band_index] yields 1-element arrays.
            vals = np.fromiter(
                (v[0] for v in ds.sample(coords, indexes=[band_index])),
                dtype="float64",
                count=int(in_bounds.sum()),
            )
            sampled[in_bounds] = vals

    calibrated = apply_calibration(sampled, scale, offset, nodata)

    # Belt-and-suspenders: any negative "radiance" after calibration is a
    # sentinel/nodata artifact — the Glimmer sensor cannot produce negative
    # physical radiance. Null those too so they fall back to VIIRS.
    calibrated = np.where(calibrated < 0, np.nan, calibrated)
    return pd.Series(calibrated, index=gdf.index, dtype="float64")


def attach_sdgsat_columns(
    gdf: gpd.GeoDataFrame,
    pan_raster: Path | None,
    mss_raster: Path | None,
    bands_cfg: dict[str, dict[str, Any]],
    radiometric_cfg: dict[str, float],
) -> gpd.GeoDataFrame:
    """Attach ``alan_sdgsat`` (pan) and ``alan_sdgsat_blue`` columns.

    Either raster may be ``None`` — the corresponding column comes back all-NaN.
    That's the same outcome as "no coverage" and downstream (S4) will fall
    back to VIIRS ``alan_radiance``.
    """
    out = gdf.copy()
    scale = float(radiometric_cfg.get("scale", 1.0))
    offset = float(radiometric_cfg.get("offset", 0.0))

    if pan_raster is not None and Path(pan_raster).exists():
        out["alan_sdgsat"] = sample_band_at_centroids(
            Path(pan_raster), gdf, int(bands_cfg["pan"]["index"]), scale, offset
        )
    else:
        out["alan_sdgsat"] = pd.Series(np.nan, index=gdf.index, dtype="float64")

    if mss_raster is not None and Path(mss_raster).exists():
        out["alan_sdgsat_blue"] = sample_band_at_centroids(
            Path(mss_raster), gdf, int(bands_cfg["blue"]["index"]), scale, offset
        )
    else:
        out["alan_sdgsat_blue"] = pd.Series(np.nan, index=gdf.index, dtype="float64")

    return out


def coverage_stats(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    """Small stats dict for the driver script to print. Coverage % = fraction
    of buildings with a non-null SDGSAT reading (i.e. they'd use SDGSAT, not
    the VIIRS fallback)."""
    n = int(len(gdf))
    stats: dict[str, Any] = {"buildings": n}
    for col in ("alan_sdgsat", "alan_sdgsat_blue"):
        if col not in gdf.columns:
            continue
        s = gdf[col]
        non_null = int(s.notna().sum())
        stats[f"{col}_covered"] = non_null
        stats[f"{col}_covered_pct"] = 100.0 * non_null / n if n else 0.0
        stats[f"{col}_median"] = float(s.median()) if non_null else None
        stats[f"{col}_max"] = float(s.max()) if non_null else None
    return stats
