"""Inspect the fetched Overture GeoJSON for the active city.

Reports:
  - total polygon count
  - available columns
  - non-null coverage for the fields we care about (height, num_floors,
    class, subtype)
  - per-height quick summary (min, median, max)

Height coverage decides the T2 backfill strategy:
    - mostly non-null   -> thin OSM levels backfill
    - mostly null       -> OSM backfill becomes the main path
    - column missing    -> lean on A + OSM only, document H-null policy
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd

from pipeline import config

INTERESTING_FIELDS = ["height", "num_floors", "class", "subtype", "roof_shape"]


def main() -> int:
    cfg = config.load()
    city = config.active_city(cfg)

    dev_path = Path(city["footprints_geojson"])
    dev_path = dev_path.with_name(dev_path.stem + "_dev" + dev_path.suffix)
    full_path = Path(city["footprints_geojson"])

    path = dev_path if dev_path.exists() else full_path
    if not path.exists():
        print(f"[FAIL] neither {dev_path} nor {full_path} exists.")
        print("       Run: uv run python scripts/fetch_overture_city.py")
        return 1

    print(f"Inspecting {path}")
    gdf = gpd.read_file(path)
    print(f"  {len(gdf):,} polygons")
    print(f"  crs = {gdf.crs}")
    print(f"  columns = {list(gdf.columns)}")

    print("\nCoverage on key fields:")
    print(f"  {'field':16s} {'present':>7s} {'non-null':>9s} {'coverage%':>10s}")
    for f in INTERESTING_FIELDS:
        if f not in gdf.columns:
            print(f"  {f:16s} {'no':>7s} {'-':>9s} {'-':>10s}")
            continue
        non_null = gdf[f].notna().sum()
        coverage = 100.0 * non_null / len(gdf) if len(gdf) else 0.0
        print(f"  {f:16s} {'yes':>7s} {non_null:>9,d} {coverage:>9.1f}%")

    if "height" in gdf.columns and gdf["height"].notna().any():
        h = gdf["height"].dropna()
        print(f"\nHeight (meters): min={h.min():.1f}  median={h.median():.1f}  "
              f"max={h.max():.1f}  n={len(h):,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
