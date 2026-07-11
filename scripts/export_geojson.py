"""Export the scored buildings GPKG to a kepler.gl-ready GeoJSON.

Reprojects to WGS84 (what kepler expects for accurate positioning), trims to
the columns useful for a demo, and coerces NaN → None so the JSON is
strictly valid (some viewers reject bare `NaN` tokens).

Output lands next to the source, with a ``.geojson`` extension.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import pandas as pd

from pipeline import config

# Columns to include in the export. Everything else drops.
EXPORT_COLUMNS = [
    "id",
    "class",
    "subtype",
    "height",
    "footprint_area",
    "perimeter",
    "facade_area",
    "alan_radiance",
    "structure_score",
    "risk_raw",
    "risk_score",
    "geometry",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use full-city gpkg instead of dev")
    args = parser.parse_args()

    cfg = config.load()
    city = config.active_city(cfg)
    tag = "full" if args.full else "dev"

    in_path = (
        Path(cfg["output"]["local_dir"])
        / "processed"
        / f"{city['name']}_buildings_{tag}_scored.gpkg"
    )
    if not in_path.exists():
        print(f"[FAIL] {in_path} not found. Run T4 first.")
        return 1

    gdf = gpd.read_file(in_path)
    print(f"Loaded {len(gdf):,} rows from {in_path.name}")

    keep = [c for c in EXPORT_COLUMNS if c in gdf.columns]
    slim = gdf[keep].copy()

    # Drop rows without a risk score — nothing to render for them.
    slim = slim.dropna(subset=["risk_raw"]).reset_index(drop=True)

    # kepler.gl wants WGS84.
    slim = slim.to_crs("EPSG:4326")

    # Build GeoJSON by hand so NaN becomes JSON null (not the literal `NaN`
    # token that geopandas' default writer emits — kepler is tolerant but
    # strict validators aren't, and this keeps the output portable).
    non_geom = [c for c in slim.columns if c != "geometry"]
    features = []
    for _, row in slim.iterrows():
        props = {}
        for col in non_geom:
            val = row[col]
            if pd.isna(val):
                props[col] = None
            elif isinstance(val, (int, float)):
                props[col] = float(val) if isinstance(val, float) else int(val)
            else:
                props[col] = val
        features.append({
            "type": "Feature",
            "geometry": row.geometry.__geo_interface__,
            "properties": props,
        })

    fc = {"type": "FeatureCollection", "features": features}

    out_path = in_path.with_suffix(".geojson")
    with open(out_path, "w") as fh:
        json.dump(fc, fh)

    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path.name}: {size_mb:.1f} MB ({len(features):,} features)")
    print(f"  path = {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
