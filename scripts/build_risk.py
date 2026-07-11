"""Compute the risk index for the ALAN-enriched buildings and save the scored layer.

Reads ``data/processed/<city>_buildings_<mode>_alan.gpkg`` (T3 output), applies
the risk model (T4), writes ``..._scored.gpkg``, and prints top-N buildings.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd

from pipeline import config, habitat, risk


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full-city gpkg instead of the dev one",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top-ranked buildings to print (default 10)",
    )
    args = parser.parse_args()

    cfg = config.load()
    city = config.active_city(cfg)
    tag = "full" if args.full else "dev"

    in_path = (
        Path(cfg["output"]["local_dir"])
        / "processed"
        / f"{city['name']}_buildings_{tag}_alan.gpkg"
    )
    if not in_path.exists():
        print(f"[FAIL] {in_path} not found. Run T3 first: uv run python scripts/build_alan.py{' --full' if args.full else ''}")
        return 1

    print(f"T4 pipeline — city={city['name']}, mode={tag}")
    print(f"  input  = {in_path}")
    gdf = gpd.read_file(in_path)

    # v2A: attach habitat-edge multiplier from OSM.
    habitat_cfg = cfg.get("habitat")
    if habitat_cfg:
        bbox = city["dev_bbox" if not args.full else "bbox"]
        print(f"  pulling habitat features from OSM (parks/water/wood) for bbox…")
        t_h = time.perf_counter()
        hab = habitat.load_habitat_features(
            bbox_wgs84=tuple(bbox),
            osm_tags=habitat_cfg["osm_tags"],
            projected_crs=city["projected_crs"],
        )
        dist = habitat.distance_to_nearest_habitat(gdf, hab)
        gdf["habitat_edge"] = habitat.edge_multiplier(dist, habitat_cfg["decay_m"])
        print(f"    {len(hab):,} habitat polygons; "
              f"median distance = {dist.replace([float('inf')], float('nan')).median():.0f} m "
              f"({time.perf_counter() - t_h:.1f}s)")

    t0 = time.perf_counter()
    scored, stats = risk.run(cfg, gdf)
    dt = time.perf_counter() - t0

    print(f"\nRisk stats (computed in {dt:.2f}s):")
    for k, v in stats.items():
        if k == "top10_ids":
            continue
        if isinstance(v, float):
            print(f"  {k:22s} {v:>12.4f}")
        else:
            print(f"  {k:22s} {v:>12}")

    top = (
        scored.dropna(subset=["risk_raw"])
        .sort_values("risk_raw", ascending=False)
        .head(args.top)
    )
    cols = [c for c in ("id", "class", "height", "footprint_area", "alan_radiance", "risk_score") if c in top.columns]
    print(f"\nTop {len(top)} by risk_raw:")
    print(top[cols].to_string(index=False))

    out_path = in_path.with_name(in_path.stem.replace("_alan", "_scored") + ".gpkg")
    scored.to_file(out_path, driver="GPKG")
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote {out_path.name}: {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
