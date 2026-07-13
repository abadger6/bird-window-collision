"""S5 — The MVP experiment: does 40 m + blue reorder the top-N vs 500 m VIIRS?

Runs pipeline/risk.py three times on the same S3 output (VIIRS + SDGSAT
columns already attached) — once per ``lighting_source`` — and compares
rankings:

    * Top-25 leaderboards, side by side, keyed on building id.
    * Spearman rank correlation of risk_score(VIIRS) vs risk_score(SDGSAT_pan)
      and vs risk_score(SDGSAT_blue).
    * Biggest movers: buildings whose rank changed most, printed with height,
      footprint, and both radiance values so you can eyeball why.
    * A one-paragraph auto-summary — the actual decision output for whether
      to invest in formal fusion (see docs/sdgsat_alan.md "Open decisions").

Writes ``data/processed/lighting_comparison.csv`` — one row per building
with rank + risk_score under each source, plus rank_delta_* columns.

Assumes S3 has been run and the habitat-edge column is on the input frame
(add if needed, mirroring scripts/build_risk.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats as spstats

from pipeline import config, habitat, risk


SOURCES = ("viirs", "sdgsat_pan", "sdgsat_blue")


def _ensure_habitat_column(gdf: gpd.GeoDataFrame, cfg: dict, city: dict, tag: str) -> gpd.GeoDataFrame:
    """S5 shares a frame across three model runs; compute habitat_edge once
    up front rather than three times inside risk.run()."""
    if "habitat_edge" in gdf.columns:
        return gdf
    habitat_cfg = cfg.get("habitat")
    if not habitat_cfg:
        return gdf
    bbox = city["dev_bbox" if tag == "dev" else "bbox"]
    print(f"  pulling habitat features (parks/water/wood) for bbox…")
    hab = habitat.load_habitat_features(
        bbox_wgs84=tuple(bbox),
        osm_tags=habitat_cfg["osm_tags"],
        projected_crs=city["projected_crs"],
    )
    dist = habitat.distance_to_nearest_habitat(gdf, hab)
    gdf["habitat_edge"] = habitat.edge_multiplier(dist, habitat_cfg["decay_m"])
    return gdf


def score_under_source(cfg: dict, gdf: pd.DataFrame, source: str) -> pd.DataFrame:
    """Run the risk index with lighting_source overridden to ``source``.
    Returns just the columns we need for comparison (id + score + rank)."""
    cfg_local = dict(cfg)
    cfg_local["lighting_source"] = source
    scored, _ = risk.run(cfg_local, gdf)
    keep = scored[["id", "risk_raw", "risk_score", "L_fallback"]].copy() \
        if "id" in scored.columns else scored[["risk_raw", "risk_score", "L_fallback"]].copy()
    if "id" not in keep.columns:
        keep = keep.reset_index().rename(columns={"index": "id"})
    keep = keep.rename(columns={
        "risk_raw":    f"risk_raw_{source}",
        "risk_score":  f"risk_score_{source}",
        "L_fallback":  f"L_fallback_{source}",
    })
    # Dense rank descending so highest risk = rank 1.
    keep[f"rank_{source}"] = (
        keep[f"risk_raw_{source}"].rank(method="min", ascending=False).astype("Int64")
    )
    return keep


def build_comparison(scored_by_src: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge the per-source frames on ``id`` and add rank_delta_* columns."""
    merged: pd.DataFrame | None = None
    for src, frame in scored_by_src.items():
        merged = frame if merged is None else merged.merge(frame, on="id", how="outer")
    assert merged is not None
    baseline = f"rank_viirs"
    for src in SOURCES:
        if src == "viirs":
            continue
        merged[f"rank_delta_{src}"] = merged[baseline] - merged[f"rank_{src}"]
    return merged


def _side_by_side_topn(
    comparison: pd.DataFrame,
    gdf_meta: pd.DataFrame,
    n: int = 25,
) -> pd.DataFrame:
    """Top-N table with one column per source showing rank + id."""
    tops: dict[str, list] = {}
    for src in SOURCES:
        col = f"rank_{src}"
        top_ids = (
            comparison.sort_values(col).head(n)[["id", col]]
            .assign(**{col: lambda d: d[col].astype("Int64")})
        )
        tops[src] = top_ids.reset_index(drop=True)
    # Stitch: rank position → id_viirs, id_sdgsat_pan, id_sdgsat_blue.
    positions = pd.DataFrame({"rank": range(1, n + 1)})
    for src in SOURCES:
        positions[f"id_{src}"] = tops[src]["id"].values
    return positions


def _biggest_movers(
    comparison: pd.DataFrame,
    meta: pd.DataFrame,
    source: str,
    n: int = 15,
) -> pd.DataFrame:
    delta_col = f"rank_delta_{source}"
    movers = comparison.reindex(
        comparison[delta_col].abs().sort_values(ascending=False).index
    ).head(n)
    cols = ["id", "rank_viirs", f"rank_{source}", delta_col]
    if "height" in meta.columns:
        movers = movers.merge(
            meta[["id", "height", "footprint_area", "alan_radiance", "alan_sdgsat", "alan_sdgsat_blue"]],
            on="id", how="left",
        )
        cols += ["height", "footprint_area", "alan_radiance"]
        cols += ["alan_sdgsat" if source == "sdgsat_pan" else "alan_sdgsat_blue"]
    return movers[cols].reset_index(drop=True)


def _auto_summary(comparison: pd.DataFrame, corrs: dict[str, float]) -> str:
    """One paragraph — the actual decision output for the fusion question.

    Verdict is driven by top-25 overlap with VIIRS, not by Spearman. Across
    ~20k buildings a Spearman of ~0.95 can still hide a top-25 leaderboard
    that reshuffles by half — and the top-N is precisely what this experiment
    was set up to answer per docs/sdgsat_alan.md.
    """
    n = int(comparison["rank_viirs"].notna().sum())
    top25_viirs = set(comparison.sort_values("rank_viirs").head(25)["id"])
    top25_pan = set(comparison.sort_values("rank_sdgsat_pan").head(25)["id"])
    top25_blue = set(comparison.sort_values("rank_sdgsat_blue").head(25)["id"])
    ov_pan = len(top25_viirs & top25_pan)
    ov_blue = len(top25_viirs & top25_blue)
    worst_overlap = min(ov_pan, ov_blue)
    if worst_overlap <= 12:      # <= half the top-25 preserved
        verdict = "materially reorders"
        recommendation = "→ formal VIIRS×SDGSAT fusion is likely worth building."
    elif worst_overlap <= 18:    # a few swaps
        verdict = "modestly reorders"
        recommendation = "→ fusion is defensible but not urgent; blue-band-only weight tweak may be enough."
    else:                        # ~unchanged
        verdict = "barely reorders"
        recommendation = "→ leaderboards agree; fusion has low expected leverage over the parallel-column MVP."
    return (
        f"SDGSAT-1 Glimmer {verdict} the top-N vs 500 m VIIRS on {n:,} scored "
        f"buildings. Top-25 overlap with VIIRS: pan={ov_pan}/25, blue={ov_blue}/25. "
        f"Spearman rank corr (all buildings): sdgsat_pan={corrs['sdgsat_pan']:+.3f}, "
        f"sdgsat_blue={corrs['sdgsat_blue']:+.3f}. {recommendation}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--top", type=int, default=25, help="Leaderboard depth (default 25)")
    parser.add_argument("--movers", type=int, default=15, help="Biggest-movers to print (default 15)")
    args = parser.parse_args()

    cfg = config.load()
    city = config.active_city(cfg)
    tag = "full" if args.full else "dev"
    data_root = Path(cfg["output"]["local_dir"])

    in_path = data_root / "processed" / f"{city['name']}_buildings_{tag}_alan_hires.gpkg"
    if not in_path.exists():
        print(f"[FAIL] {in_path} not found. Run S3 first: "
              f"uv run python scripts/build_alan_hires.py{' --full' if args.full else ''}")
        return 1

    print(f"S5 comparison — city={city['name']}, mode={tag}")
    print(f"  input   = {in_path}")
    gdf = gpd.read_file(in_path)
    if "id" not in gdf.columns:
        gdf["id"] = [f"row_{i}" for i in range(len(gdf))]
    print(f"  loaded {len(gdf):,} buildings, crs={gdf.crs}")

    gdf = _ensure_habitat_column(gdf, cfg, city, tag)

    # 1) Score under each source. viirs first (baseline).
    scored_by_src: dict[str, pd.DataFrame] = {}
    for src in SOURCES:
        t0 = time.perf_counter()
        scored_by_src[src] = score_under_source(cfg, gdf, src)
        dt = time.perf_counter() - t0
        fb = int(scored_by_src[src][f"L_fallback_{src}"].sum())
        print(f"  scored under lighting_source={src:11s}  "
              f"({dt:.2f}s, {fb:,} L-fallbacks to VIIRS)")

    comparison = build_comparison(scored_by_src)

    # 2) Spearman rank correlation of risk_score(viirs) vs each SDGSAT variant.
    def _spearman(a: str, b: str) -> float:
        joined = comparison.dropna(subset=[f"risk_score_{a}", f"risk_score_{b}"])
        if len(joined) < 2:
            return float("nan")
        return float(spstats.spearmanr(
            joined[f"risk_score_{a}"], joined[f"risk_score_{b}"]
        ).correlation)

    corrs = {src: _spearman("viirs", src) for src in ("sdgsat_pan", "sdgsat_blue")}

    # 3) Top-N leaderboard side-by-side.
    print(f"\nTop {args.top} by risk_raw, side-by-side (VIIRS | SDGSAT pan | SDGSAT blue):")
    sxs = _side_by_side_topn(comparison, gdf, n=args.top)
    print(sxs.to_string(index=False))

    # 4) Biggest movers (per SDGSAT variant).
    for src in ("sdgsat_pan", "sdgsat_blue"):
        print(f"\nBiggest rank movers under lighting_source={src} "
              f"(top {args.movers} by |rank_delta|):")
        movers = _biggest_movers(comparison, gdf, src, n=args.movers)
        print(movers.to_string(index=False))

    # 5) Auto-summary.
    print("\n" + "=" * 78)
    print(_auto_summary(comparison, corrs))
    print("=" * 78)

    # 6) Persist.
    out_path = data_root / "processed" / "lighting_comparison.csv"
    # Merge in a small meta slice so the CSV is self-contained.
    meta_cols = [c for c in ("id", "height", "footprint_area", "class",
                             "alan_radiance", "alan_sdgsat", "alan_sdgsat_blue") if c in gdf.columns]
    out_df = comparison.merge(gdf[meta_cols], on="id", how="left")
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path.name}: {out_path.stat().st_size / 1e3:.1f} KB, "
          f"{len(out_df):,} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
