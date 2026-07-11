"""Sample Black Marble radiance onto the processed buildings layer.

Reads ``data/processed/<city>_buildings_<mode>.gpkg`` produced by T2, calls
GEE to compute the migration-months mean composite, samples per-centroid,
and writes ``data/processed/<city>_buildings_<mode>_alan.gpkg`` for T4.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee  # type: ignore[import-untyped]
import geopandas as gpd

from pipeline import alan, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full-city buildings gpkg instead of the dev one",
    )
    args = parser.parse_args()

    cfg = config.load()
    project = config.require_gee_project(cfg)
    city = config.active_city(cfg)

    tag = "full" if args.full else "dev"
    in_path = Path(cfg["output"]["local_dir"]) / "processed" / f"{city['name']}_buildings_{tag}.gpkg"
    if not in_path.exists():
        print(f"[FAIL] {in_path} not found. Run T2 first: uv run python scripts/build_buildings.py{' --full' if args.full else ''}")
        return 1

    print(f"T3 pipeline — city={city['name']}, mode={tag}")
    print(f"  input  = {in_path}")
    gdf = gpd.read_file(in_path)
    print(f"  loaded {len(gdf):,} buildings, crs={gdf.crs}")

    ee.Initialize(project=project)

    t0 = time.perf_counter()
    enriched, stats = alan.run(cfg, gdf)
    dt = time.perf_counter() - t0

    print(f"\nSampling stats (sampled in {dt:.1f}s):")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:24s} {v:>12.3f}")
        else:
            print(f"  {k:24s} {v:>12}")

    if stats["buildings_sampled"]:
        pct_shared = 100.0 * (1 - stats["unique_radiance_values"] / stats["buildings_sampled"])
        print(f"  {'buildings sharing pixel':24s} {pct_shared:>11.1f}%   (ALAN 500 m/pixel caveat)")

    out_path = in_path.with_name(in_path.stem + "_alan.gpkg")
    enriched.to_file(out_path, driver="GPKG")
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote {out_path.name}: {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
