"""Run a lighting-reduction scenario on the scored buildings and print the impact.

Demo/reference CLI for T5. By default treats the top-N risk-ranked buildings
with the given reduction. Use ``--ids`` to treat a specific list instead.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd

from pipeline import config, scenario


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full-city scored gpkg instead of dev",
    )
    parser.add_argument(
        "--reduction",
        type=float,
        default=0.5,
        help="Lighting-reduction fraction in [0, 1] (default 0.5 = 50%%)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Treat the top-N highest-risk buildings (ignored if --ids given)",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="Explicit list of building IDs to treat (overrides --top)",
    )
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
        print(f"[FAIL] {in_path} not found. Run T4 first: uv run python scripts/build_risk.py{' --full' if args.full else ''}")
        return 1

    gdf = gpd.read_file(in_path)

    if args.ids:
        treated_ids = args.ids
        selection_desc = f"explicit ({len(treated_ids)} ids)"
    else:
        treated_ids = (
            gdf.dropna(subset=["risk_raw"])
            .sort_values("risk_raw", ascending=False)
            .head(args.top)["id"]
            .tolist()
        )
        selection_desc = f"top {args.top} by risk_raw"

    result = scenario.apply_lighting_reduction(gdf, treated_ids, args.reduction)

    print(f"Scenario — city={city['name']}, mode={tag}")
    print(f"  treated_selection      {selection_desc}")
    print(f"  reduction_fraction     {result.reduction_fraction:>10.2f}")
    print(f"  risk_total_before      {result.risk_total_before:>10.4f}")
    print(f"  risk_total_after       {result.risk_total_after:>10.4f}")
    print(f"  risk_removed           {result.risk_removed:>10.4f}")
    pct = 100 * result.risk_removed / result.risk_total_before if result.risk_total_before else 0.0
    print(f"  % of city risk removed {pct:>10.2f}%")
    print(f"\nTop treated buildings by delta:")
    print(result.treated_deltas.head(min(10, len(result.treated_deltas))).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
