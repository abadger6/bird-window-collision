"""Fetch Overture buildings for the active city's dev_bbox to a local GeoJSON.

Uses the official ``overturemaps`` package, which resolves the latest release
version and pulls straight from Overture's S3 bucket. No auth needed.

Writes to ``<repo>/<cities.<active>.footprints_geojson>``. Overwrites.

By default we fetch the dev_bbox (a few thousand polygons) so the T2 pipeline
is iterable. Pass ``--full`` to fetch the full city bbox (~500k+ polygons for
Chicago — slower and larger).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from overturemaps import core as overture_core, geodataframe

from pipeline import config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Fetch the full city bbox instead of dev_bbox",
    )
    args = parser.parse_args()

    cfg = config.load()
    city = config.active_city(cfg)
    bbox_key = "bbox" if args.full else "dev_bbox"
    bbox = tuple(city[bbox_key])

    out_path = Path(city["footprints_geojson"])
    if not args.full:
        out_path = out_path.with_name(out_path.stem + "_dev" + out_path.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    release = overture_core.get_latest_release()
    print(f"Fetching Overture buildings")
    print(f"  city    = {city['name']}")
    print(f"  bbox    = {bbox}  ({'full' if args.full else 'dev'})")
    print(f"  release = {release}")
    print(f"  output  = {out_path}")

    t0 = time.perf_counter()
    gdf = geodataframe("building", bbox=bbox)
    dt_fetch = time.perf_counter() - t0
    print(f"  fetched {len(gdf):,} polygons in {dt_fetch:.1f}s")

    # Overture returns geometry in WGS84 but doesn't stamp a CRS on the frame.
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    t1 = time.perf_counter()
    gdf.to_file(out_path, driver="GeoJSON")
    dt_write = time.perf_counter() - t1

    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {out_path.name}: {size_mb:.1f} MB in {dt_write:.1f}s")
    print("Next: uv run python scripts/inspect_local_footprints.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
