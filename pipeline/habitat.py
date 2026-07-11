"""Model v2A — habitat-edge multiplier.

Bird collision risk concentrates within a few hundred meters of habitat edges
(parks, water, wooded areas). Buildings adjacent to greenspace should receive
a risk multiplier reflecting that concentration.

    E_i = exp(-distance_to_nearest_habitat / decay_m)     (in [0, 1])
    Risk_raw_i *= (1 + w_E * E_i)                          (§5.2 with §5.1 add)

Habitat features come from OSM via ``osmnx``. Tag list is configurable.
Distances are computed in the city's projected CRS (meters).
"""
from __future__ import annotations

from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd


def load_habitat_features(
    bbox_wgs84: tuple[float, float, float, float],
    osm_tags: list[dict[str, Any]] | dict[str, Any],
    projected_crs: str,
) -> gpd.GeoDataFrame:
    """Fetch parks/water/wood polygons from OSM for the bbox.

    ``osm_tags`` may be either a single osmnx-tags dict or a list of them
    (which we merge — osmnx doesn't take a list, so we OR them together).
    """
    if isinstance(osm_tags, list):
        merged: dict[str, Any] = {}
        for tag in osm_tags:
            for k, v in tag.items():
                if k in merged:
                    prev = merged[k] if isinstance(merged[k], list) else [merged[k]]
                    add = v if isinstance(v, list) else [v]
                    merged[k] = prev + add
                else:
                    merged[k] = v
        osm_tags = merged

    minx, miny, maxx, maxy = bbox_wgs84
    # osmnx 2.x uses (left, bottom, right, top) = (minx, miny, maxx, maxy)
    gdf = ox.features.features_from_bbox(
        bbox=(minx, miny, maxx, maxy),
        tags=osm_tags,
    )
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=projected_crs)

    # Keep polygonal geometries only — nearest-distance is defined for lines
    # and points too, but for habitat proximity we want areal features.
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf.to_crs(projected_crs)
    return gdf


def distance_to_nearest_habitat(
    buildings: gpd.GeoDataFrame,
    habitat: gpd.GeoDataFrame,
) -> pd.Series:
    """Return distance in meters from each building centroid to nearest habitat polygon.

    If ``habitat`` is empty, returns +inf for every building (so E=0 and the
    multiplier is a no-op).
    """
    if habitat.empty:
        return pd.Series(np.inf, index=buildings.index, name="habitat_distance_m")

    centroids = gpd.GeoDataFrame(
        geometry=buildings.geometry.centroid,
        crs=buildings.crs,
    )
    joined = gpd.sjoin_nearest(
        centroids,
        habitat[["geometry"]],
        distance_col="habitat_distance_m",
        how="left",
    )
    # sjoin_nearest can produce duplicate rows when multiple habitat features
    # tie for closest; keep the first per building.
    joined = joined[~joined.index.duplicated(keep="first")]
    return joined["habitat_distance_m"].reindex(buildings.index)


def edge_multiplier(distance_m: pd.Series, decay_m: float) -> pd.Series:
    """Exponential decay from habitat distance to a [0, 1] multiplier."""
    if decay_m <= 0:
        raise ValueError(f"decay_m must be positive, got {decay_m}")
    return np.exp(-distance_m / decay_m).astype("float64")
