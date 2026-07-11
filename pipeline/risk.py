"""T4 — Relative bird–window collision risk index (v2).

Model as of v2:

    H_effective_i = sigmoid((height_i - midpoint_m) / slope_m)     # C: nonlinear H
    Structure_i   = w_F·norm(F_i) + w_A·norm(A_i) + w_H·norm(H_eff_i)
    L_class_i     = norm(L_i) * class_multipliers[class_i]         # B: class weighting
    Risk_raw_i    = Structure_i * L_class_i * (1 + w_E * E_i)      # A: habitat edge
    Risk_i        = 100 * percentile_rank(Risk_raw_i)               # display only

Where:
    F = perimeter * height    (physical facade area — NOT sigmoid-transformed)
    A = footprint area
    H = building height (m)
    L = mean ALAN radiance from Black Marble
    E = exp(-distance_to_nearest_habitat / decay_m)   ∈ (0, 1]

Scenario math (T5) operates on ``risk_raw``; the class multiplier and E
factor are baked in there because they're per-building constants.

Null handling: for a building missing structural terms (typically because
``height`` was null), Structure is computed on the subset of terms present
with weights renormalized. A building missing every term produces NaN.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

STRUCTURE_WEIGHT_KEYS = ("facade", "area", "height")
STRUCTURE_COLUMN_MAP = {
    "facade": "facade_area",
    "area": "footprint_area",
    "height": "height_effective",  # sigmoid-transformed H — see apply_height_response
}


def min_max(series: pd.Series) -> pd.Series:
    """Min–max normalize to [0, 1]. Nulls pass through. Constant series -> 0."""
    valid = series.dropna()
    if valid.empty:
        return series.astype("float64")
    lo, hi = valid.min(), valid.max()
    if hi == lo:
        return series.where(series.isna(), 0.0).astype("float64")
    return ((series - lo) / (hi - lo)).astype("float64")


def apply_height_response(height_m: pd.Series, midpoint_m: float, slope_m: float) -> pd.Series:
    """Sigmoid transform of height (§C — nonlinear H response).

    Preserves nulls. Grounded in Loss 2014 / Wang 2020: collision counts rise
    sharply above ~30m and plateau at very tall heights.
    """
    if slope_m <= 0:
        raise ValueError(f"slope_m must be positive, got {slope_m}")
    z = (height_m.astype("float64") - float(midpoint_m)) / float(slope_m)
    return 1.0 / (1.0 + np.exp(-z))


def _validate_weights(weights: dict[str, float]) -> None:
    total = sum(weights[k] for k in STRUCTURE_WEIGHT_KEYS)
    if not (0.999 <= total <= 1.001):
        raise ValueError(
            f"Structure weights must sum to 1.0 (got {total:.4f}). "
            f"Fix config.yaml → weights.structure."
        )


def _structure_row(row: pd.Series, weights: dict[str, float]) -> float:
    """Partial-score with weight renormalization over available terms."""
    parts: list[tuple[float, float]] = []
    for key in STRUCTURE_WEIGHT_KEYS:
        val = row.get(f"norm_{STRUCTURE_COLUMN_MAP[key]}")
        if val is not None and not pd.isna(val):
            parts.append((weights[key], float(val)))
    if not parts:
        return np.nan
    total_weight = sum(w for w, _ in parts)
    return sum(w * v for w, v in parts) / total_weight


def compute(
    gdf: pd.DataFrame,
    weights: dict[str, float],
    height_response: dict[str, float] | None = None,
    class_multipliers: dict[str, float] | None = None,
    edge_weight: float = 0.0,
) -> pd.DataFrame:
    """Attach norm_*, structure_score, risk_raw, and risk_score columns.

    ``height_response``: {"midpoint_m": float, "slope_m": float} — enables §C.
        If None, falls back to raw H (v1 behavior).
    ``class_multipliers``: dict class → multiplier — enables §B.
        Any missing class or null falls back to 1.0.
    ``edge_weight``: w_E in [0, +∞). Requires ``habitat_edge`` column on the
        input frame (populated by pipeline/habitat.py). If 0 or column
        missing, the edge term collapses to a no-op.
    """
    _validate_weights(weights)
    out = gdf.copy()

    # §C — apply sigmoid to height, then normalize the transformed series
    if height_response is not None and "height" in out.columns:
        out["height_effective"] = apply_height_response(
            out["height"], height_response["midpoint_m"], height_response["slope_m"]
        )
    else:
        out["height_effective"] = out["height"] if "height" in out.columns else np.nan

    for col in ("facade_area", "footprint_area", "height_effective", "alan_radiance"):
        if col not in out.columns:
            raise KeyError(f"expected column {col!r} in input; run T2/T3 first")
        out[f"norm_{col}"] = min_max(out[col])

    out["structure_score"] = out.apply(_structure_row, axis=1, args=(weights,))

    # §B — class multiplier on norm(L)
    if class_multipliers and "class" in out.columns:
        class_mult = out["class"].map(class_multipliers).fillna(1.0)
    else:
        class_mult = pd.Series(1.0, index=out.index)
    out["class_multiplier"] = class_mult.astype("float64")

    # §A — habitat edge multiplier
    if edge_weight > 0 and "habitat_edge" in out.columns:
        edge_factor = 1.0 + edge_weight * out["habitat_edge"].astype("float64")
    else:
        edge_factor = pd.Series(1.0, index=out.index)
    out["edge_factor"] = edge_factor

    out["risk_raw"] = (
        out["structure_score"]
        * out["norm_alan_radiance"]
        * out["class_multiplier"]
        * out["edge_factor"]
    )
    out["risk_score"] = out["risk_raw"].rank(pct=True) * 100.0
    return out


def run(cfg: dict[str, Any], gdf: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full T4 pipeline. Returns the scored frame and a small stats dict."""
    weights = cfg["weights"]["structure"]
    height_response = cfg.get("height_response")
    class_multipliers = cfg.get("class_multipliers") or {}
    edge_weight = float(cfg["weights"].get("edge", 0.0))

    scored = compute(
        gdf,
        weights=weights,
        height_response=height_response,
        class_multipliers=class_multipliers,
        edge_weight=edge_weight,
    )

    valid = int(scored["risk_raw"].notna().sum())
    top10 = (
        scored.dropna(subset=["risk_raw"])
        .sort_values("risk_raw", ascending=False)
        .head(10)
    )

    stats = {
        "buildings_scored": int(len(scored)),
        "risk_non_null": valid,
        "risk_null": int(len(scored) - valid),
        "risk_raw_min": float(scored["risk_raw"].min()) if valid else None,
        "risk_raw_median": float(scored["risk_raw"].median()) if valid else None,
        "risk_raw_max": float(scored["risk_raw"].max()) if valid else None,
        "top10_ids": top10["id"].tolist() if "id" in top10.columns else [],
    }
    return scored, stats
