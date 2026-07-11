"""T3 — Nighttime radiance sampling from NASA Black Marble VNP46A2.

For each building row, attaches ``alan_radiance``: the mean nighttime
radiance at the building's centroid, averaged across the migration-months
window configured in ``alan_composite``. This becomes the L term in the
risk index (T4).

Design notes:

* Sampled at **centroid**, not polygon interior. VNP46A2 is 500 m/pixel;
  Chicago's median building is ~500 m², so polygon-mean at 500 m scale
  drops most small buildings to null. Centroid sampling gives every
  building exactly one value — at the cost of many buildings sharing a
  value because they share a pixel. That's the "district-scale lighting
  environment" caveat from plan §4.2 made explicit.

* Server-side composite + ``sampleRegions``. Same pattern scales from
  dev bbox to full city without a rewrite. Full city needs batching to
  avoid ``getInfo`` limits — deferred until we run at that scale.

* No QA masking. ``Gap_Filled_DNB_BRDF_Corrected_NTL`` already handles
  most QA issues; explicit QA mask adds complexity for marginal gain.
"""
from __future__ import annotations

from typing import Any

import ee  # type: ignore[import-untyped]
import geopandas as gpd
import pandas as pd


def build_composite(cfg: dict[str, Any]) -> ee.Image:
    """Mean-radiance composite across the configured migration months/years."""
    alan_id = cfg["assets"]["alan"]
    band = cfg["assets"]["alan_band"]
    years = cfg["alan_composite"]["years"]
    months = cfg["alan_composite"]["months"]

    ic = ee.ImageCollection(alan_id).select(band)
    year_filter = ee.Filter.calendarRange(min(years), max(years), "year")
    month_filters = [ee.Filter.calendarRange(m, m, "month") for m in months]
    season_filter = ee.Filter.Or(*month_filters) if len(month_filters) > 1 else month_filters[0]

    filtered = ic.filter(year_filter).filter(season_filter)
    # Rename to a stable property name so callers don't need to know the band id.
    return filtered.mean().rename("alan")


DEFAULT_BATCH_SIZE = 1000


def sample_at_centroids(
    image: ee.Image,
    gdf: gpd.GeoDataFrame,
    scale: int = 500,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> gpd.GeoDataFrame:
    """Attach ``alan_radiance`` per building by sampling ``image`` at centroids.

    ``gdf`` may be in any CRS; centroids are computed in ``gdf.crs`` then
    reprojected to WGS84 for GEE.

    Batches the client→server round-trip so we stay under GEE's practical
    ``getInfo`` size limit (~5000 features per call). For full-city work at
    500k centroids this becomes the difference between success and cryptic
    timeouts.
    """
    if gdf.crs is None:
        raise ValueError("gdf must have a CRS set")

    centroids_wgs = gdf.geometry.centroid.to_crs("EPSG:4326")
    idx_values = list(gdf.index)
    n = len(idx_values)
    values: dict[int, float] = {}

    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        batch_features = [
            ee.Feature(ee.Geometry.Point([float(pt.x), float(pt.y)]), {"idx": int(idx)})
            for idx, pt in zip(idx_values[start:stop], centroids_wgs.iloc[start:stop])
        ]
        fc = ee.FeatureCollection(batch_features)
        sampled = image.sampleRegions(collection=fc, scale=scale, geometries=False)
        payload = sampled.getInfo()["features"]
        for feat in payload:
            props = feat.get("properties", {})
            idx = props.get("idx")
            val = props.get("alan")
            if idx is not None:
                values[int(idx)] = val
        if n > batch_size:
            print(f"  ALAN batch {start // batch_size + 1}/{(n - 1) // batch_size + 1}: "
                  f"{stop:,}/{n:,} centroids sampled")

    out = gdf.copy()
    out["alan_radiance"] = pd.Series(values, dtype="float64").reindex(out.index)
    return out


def run(cfg: dict[str, Any], gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Full T3 pipeline. Returns the enriched GeoDataFrame and coverage stats.

    Assumes ``ee.Initialize(project=...)`` has already been called by the caller.
    """
    image = build_composite(cfg)
    enriched = sample_at_centroids(image, gdf)

    radiance = enriched["alan_radiance"]
    stats = {
        "buildings_sampled": int(len(enriched)),
        "radiance_non_null": int(radiance.notna().sum()),
        "radiance_null": int(radiance.isna().sum()),
        "unique_radiance_values": int(radiance.dropna().nunique()),
        "radiance_min": float(radiance.min()) if radiance.notna().any() else None,
        "radiance_median": float(radiance.median()) if radiance.notna().any() else None,
        "radiance_max": float(radiance.max()) if radiance.notna().any() else None,
    }
    return enriched, stats
