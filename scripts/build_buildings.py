"""Run the T2 building pipeline for the active city.

Reads the Overture GeoJSON, cleans + backfills heights + computes geometry
terms, then writes a processed GeoPackage that T3 will consume. Also prints
coverage stats.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import buildings, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full-city Overture fetch instead of the dev bbox",
    )
    args = parser.parse_args()

    cfg = config.load()
    city = config.active_city(cfg)

    print(f"T2 pipeline — city={city['name']}, mode={'full' if args.full else 'dev'}")
    t0 = time.perf_counter()
    gdf, stats = buildings.run(cfg, dev=not args.full)
    dt = time.perf_counter() - t0

    print(f"\nCoverage stats:")
    for k, v in stats.items():
        print(f"  {k:38s} {v:>10,d}")
    total = stats["final_polygons"]
    if total:
        pct = 100.0 * stats["height_after_backfill_and_filter"] / total
        print(f"  height coverage after backfill          {pct:>9.1f}%")

    out_dir = Path(cfg["output"]["local_dir"]) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "full" if args.full else "dev"
    out_path = out_dir / f"{city['name']}_buildings_{tag}.gpkg"
    gdf.to_file(out_path, driver="GPKG")

    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote {out_path.name}: {size_mb:.1f} MB in {dt:.1f}s total pipeline time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
