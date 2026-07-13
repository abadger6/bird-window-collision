"""S3 — Attach SDGSAT-1 Glimmer radiance to the ALAN-enriched buildings layer.

Reads the T3 output (``data/processed/<city>_buildings_<mode>_alan.gpkg``)
plus the S1 mosaics (``data/sdgsat/<city>/<city>_sdgsat_{pan,mss}_mosaic.tif``),
samples pan + blue radiance per building centroid via pipeline/alan_hires.py,
and writes ``..._alan_hires.gpkg`` with both the VIIRS ``alan_radiance`` and
the new ``alan_sdgsat`` / ``alan_sdgsat_blue`` columns.

Logs:
  - % buildings covered by SDGSAT (each band)
  - correlation between VIIRS and SDGSAT radiance (covered subset only)
  - how many would fall back to VIIRS under the S4 lighting_source switch
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import numpy as np

from pipeline import alan_hires, config


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1e4 else f"{v:.1f}"
    return str(v)


def _correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson r on the co-covered subset. None if <2 shared observations."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    cfg = config.load()
    city = config.active_city(cfg)
    tag = "full" if args.full else "dev"
    data_root = Path(cfg["output"]["local_dir"])

    in_path = data_root / "processed" / f"{city['name']}_buildings_{tag}_alan.gpkg"
    if not in_path.exists():
        print(f"[FAIL] {in_path} not found. Run T3 first: "
              f"uv run python scripts/build_alan.py{' --full' if args.full else ''}")
        return 1

    scene_dir = data_root / "sdgsat" / city["name"]
    pan_mosaic = scene_dir / f"{city['name']}_sdgsat_pan_mosaic.tif"
    mss_mosaic = scene_dir / f"{city['name']}_sdgsat_mss_mosaic.tif"
    if not (pan_mosaic.exists() or mss_mosaic.exists()):
        print(f"[FAIL] No SDGSAT mosaic found in {scene_dir}.")
        print("       Download scenes, then run: uv run python scripts/fetch_sdgsat.py")
        return 1

    print(f"S3 pipeline — city={city['name']}, mode={tag}")
    print(f"  input   = {in_path}")
    print(f"  pan     = {pan_mosaic if pan_mosaic.exists() else '(none)'}")
    print(f"  mss     = {mss_mosaic if mss_mosaic.exists() else '(none)'}")

    gdf = gpd.read_file(in_path)
    print(f"  loaded {len(gdf):,} buildings, crs={gdf.crs}")

    hires_cfg = cfg["assets"]["alan_hires"]
    t0 = time.perf_counter()
    enriched = alan_hires.attach_sdgsat_columns(
        gdf,
        pan_raster=pan_mosaic if pan_mosaic.exists() else None,
        mss_raster=mss_mosaic if mss_mosaic.exists() else None,
        bands_cfg=hires_cfg["bands"],
        radiometric_cfg=hires_cfg["radiometric"],
    )
    dt = time.perf_counter() - t0

    stats = alan_hires.coverage_stats(enriched)
    print(f"\nCoverage (sampled in {dt:.1f}s):")
    for k, v in stats.items():
        print(f"  {k:32s} {_fmt(v):>14s}")

    # Correlation of SDGSAT vs VIIRS on the covered subset — a rough sanity
    # check that both sensors agree on the "dark side vs bright side" gross
    # pattern before we start reordering rankings.
    viirs = enriched["alan_radiance"].to_numpy()
    for col in ("alan_sdgsat", "alan_sdgsat_blue"):
        if col in enriched.columns and enriched[col].notna().any():
            r = _correlation(viirs, enriched[col].to_numpy())
            print(f"  corr(VIIRS, {col:20s}) = {'n/a' if r is None else f'{r:+.3f}'}")

    # S4 fallback preview: how many buildings would fall back to VIIRS if we
    # switched lighting_source to sdgsat_pan / sdgsat_blue right now.
    n = len(enriched)
    for src, col in (("sdgsat_pan", "alan_sdgsat"), ("sdgsat_blue", "alan_sdgsat_blue")):
        if col not in enriched.columns:
            continue
        fallback = int(enriched[col].isna().sum())
        print(f"  under lighting_source={src:11s}: "
              f"{fallback:,}/{n:,} buildings would fall back to VIIRS "
              f"({100.0 * fallback / n:.1f}%)")

    out_path = in_path.with_name(in_path.stem + "_hires.gpkg")
    enriched.to_file(out_path, driver="GPKG")
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nWrote {out_path.name}: {size_mb:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
