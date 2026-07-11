"""T2 — Building data pipeline.

Reads the locally-fetched Overture GeoJSON (produced by
``scripts/fetch_overture_city.py``) and produces a cleaned GeoDataFrame with
the geometry terms that feed the risk model:

    footprint_area (m^2)  — A term
    perimeter      (m)    — used to derive facade
    height         (m)    — H term (nullable)
    facade_area    (m^2)  — F term (nullable where height is null)

Height sources, in order of preference:
    1. Overture ``height``          (LiDAR-derived in the US)
    2. Overture ``num_floors * 3m`` (used to backfill when ``height`` is null)
    3. still null                    (row is kept; H/F terms are null for it)

OSM ``building:levels`` backfill is intentionally deferred — Overture +
``num_floors`` gets us to ~80% coverage in Chicago, and the OSM step adds
another network dep for a marginal gain. Add it if a target city needs it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd

# Central defaults. Overridable per call for tests / other cities.
DEFAULT_FLOOR_HEIGHT_M = 3.0
DEFAULT_MIN_AREA_M2 = 20.0
DEFAULT_MIN_HEIGHT_M = 2.0
DEFAULT_MAX_HEIGHT_M = 500.0


def load_footprints(city_cfg: dict[str, Any], dev: bool = True) -> gpd.GeoDataFrame:
    """Load the raw Overture GeoJSON for the active city."""
    path = Path(city_cfg["footprints_geojson"])
    if dev:
        path = path.with_name(path.stem + "_dev" + path.suffix)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: uv run python scripts/fetch_overture_city.py"
            f"{' --full' if not dev else ''}"
        )
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def clean_footprints(
    gdf: gpd.GeoDataFrame,
    projected_crs: str,
    min_area_m2: float = DEFAULT_MIN_AREA_M2,
) -> gpd.GeoDataFrame:
    """Reproject to a meters CRS, repair invalid geoms, drop degenerate ones."""
    gdf = gdf.to_crs(projected_crs)
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
    keep = gdf.geometry.area >= min_area_m2
    return gdf.loc[keep].reset_index(drop=True)


def fill_heights_from_num_floors(
    gdf: gpd.GeoDataFrame,
    floor_height_m: float = DEFAULT_FLOOR_HEIGHT_M,
) -> tuple[gpd.GeoDataFrame, int]:
    """Backfill null heights from num_floors × floor_height_m. Returns (gdf, filled_count)."""
    if "num_floors" not in gdf.columns:
        return gdf, 0
    mask = gdf["height"].isna() & gdf["num_floors"].notna()
    filled = int(mask.sum())
    gdf.loc[mask, "height"] = gdf.loc[mask, "num_floors"].astype(float) * floor_height_m
    return gdf, filled


def filter_implausible_heights(
    gdf: gpd.GeoDataFrame,
    min_height_m: float = DEFAULT_MIN_HEIGHT_M,
    max_height_m: float = DEFAULT_MAX_HEIGHT_M,
) -> gpd.GeoDataFrame:
    """Null out heights that are non-positive or absurdly tall. Rows remain."""
    if "height" not in gdf.columns:
        return gdf
    bad = gdf["height"].notna() & (
        (gdf["height"] < min_height_m) | (gdf["height"] > max_height_m)
    )
    gdf.loc[bad, "height"] = None
    return gdf


def compute_geometry_terms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add footprint_area, perimeter, facade_area. Assumes a projected CRS in meters."""
    gdf = gdf.copy()
    gdf["footprint_area"] = gdf.geometry.area
    gdf["perimeter"] = gdf.geometry.length
    if "height" in gdf.columns:
        gdf["facade_area"] = gdf["perimeter"] * gdf["height"]
    else:
        gdf["facade_area"] = None
    return gdf


def run(cfg: dict[str, Any], dev: bool = True) -> tuple[gpd.GeoDataFrame, dict[str, int]]:
    """Full T2 pipeline. Returns the processed GeoDataFrame and coverage stats."""
    from pipeline import config as _config

    city = _config.active_city(cfg)

    raw = load_footprints(city, dev=dev)
    raw_count = len(raw)
    overture_height_count = int(raw["height"].notna().sum()) if "height" in raw.columns else 0

    cleaned = clean_footprints(raw, city["projected_crs"])
    cleaned_count = len(cleaned)

    filled, from_num_floors = fill_heights_from_num_floors(cleaned)
    filled = filter_implausible_heights(filled)
    final = compute_geometry_terms(filled)

    height_final = int(final["height"].notna().sum()) if "height" in final.columns else 0
    height_null = len(final) - height_final

    stats = {
        "raw_polygons": raw_count,
        "cleaned_polygons": cleaned_count,
        "height_from_overture_or_kept": overture_height_count,
        "height_from_num_floors": from_num_floors,
        "height_after_backfill_and_filter": height_final,
        "height_still_null": height_null,
        "final_polygons": len(final),
    }
    return final, stats
