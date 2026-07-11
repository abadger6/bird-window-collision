"""T5 — Lighting-reduction scenario engine.

Given a scored buildings frame (T4 output), a set of treated building IDs, and
a reduction fraction ``r`` in [0, 1], recompute ``risk_raw`` under the
treatment and report:

    - total city risk_raw before / after / removed
    - per-treated-building delta (for the efficiency ranking)
    - untreated buildings' risk_raw is preserved exactly (see note below)

Design note — frozen L range. We use the *pre-treatment* L_min / L_max when
renormalizing L', rather than recomputing city-wide min/max on the scenario
frame. If we recomputed, treating a subset of buildings would shift every
untreated building's ``norm(L)`` (because the max moves) and thus their
``risk_raw`` — mathematically defensible for a strictly relative index, but
confusing in the UI ("why did this untreated building change?"). Freezing
the range keeps untreated deltas at exactly 0, which matches the policy
story: reducing lighting here removes risk here.

Scenario math operates on ``risk_raw`` (not the percentile-ranked
``risk_score``) — see §5.3 of bird-collision-risk-tool-plan.md for why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pipeline import risk


@dataclass
class ScenarioResult:
    """Outputs of a single scenario run. Everything downstream reads from this."""
    treated_ids: list[str]
    reduction_fraction: float
    risk_total_before: float
    risk_total_after: float
    risk_removed: float
    treated_deltas: pd.DataFrame = field(repr=False)


def _l_norm_frozen(l_raw: pd.Series, l_min: float, l_max: float) -> pd.Series:
    """Normalize L to [0, 1] using pre-treatment min/max. Clip out-of-range."""
    if l_max == l_min:
        return l_raw.where(l_raw.isna(), 0.0).astype("float64")
    scaled = (l_raw - l_min) / (l_max - l_min)
    return scaled.clip(lower=0.0, upper=1.0)


def apply_lighting_reduction(
    scored: pd.DataFrame,
    treated_ids: list[str] | set[str],
    reduction_fraction: float,
    id_column: str = "id",
) -> ScenarioResult:
    """Recompute ``risk_raw`` with ``L' = L × (1 − r)`` for treated buildings.

    ``scored`` must be the T4 output (has ``structure_score`` and
    ``alan_radiance`` columns). ``reduction_fraction`` in [0, 1]; 0.3 = 30%.
    """
    if not (0.0 <= reduction_fraction <= 1.0):
        raise ValueError(
            f"reduction_fraction must be in [0, 1], got {reduction_fraction}"
        )
    for col in (id_column, "structure_score", "alan_radiance", "risk_raw"):
        if col not in scored.columns:
            raise KeyError(f"expected column {col!r}; run T4 first")

    treated_set = set(treated_ids)
    treated_mask = scored[id_column].isin(treated_set)

    l_pre = scored["alan_radiance"].astype("float64")
    l_min_pre, l_max_pre = float(l_pre.min()), float(l_pre.max())

    l_post = l_pre.where(~treated_mask, l_pre * (1.0 - reduction_fraction))
    norm_l_post = _l_norm_frozen(l_post, l_min_pre, l_max_pre)

    risk_before = scored["risk_raw"].astype("float64")
    risk_after = scored["structure_score"].astype("float64") * norm_l_post

    delta = risk_before - risk_after

    treated = scored.loc[treated_mask, [id_column]].copy()
    treated["risk_raw_before"] = risk_before[treated_mask].values
    treated["risk_raw_after"] = risk_after[treated_mask].values
    treated["delta"] = delta[treated_mask].values
    treated = treated.sort_values("delta", ascending=False).reset_index(drop=True)

    return ScenarioResult(
        treated_ids=sorted(treated_set),
        reduction_fraction=float(reduction_fraction),
        risk_total_before=float(np.nansum(risk_before)),
        risk_total_after=float(np.nansum(risk_after)),
        risk_removed=float(np.nansum(delta)),
        treated_deltas=treated,
    )


def ids_within_polygon(gdf, polygon, id_column: str = "id") -> list[str]:
    """IDs of buildings whose centroid falls inside ``polygon``.

    Both ``gdf`` and ``polygon`` must share a CRS; centroids computed in that CRS.
    Returned in the frame's row order (not sorted) so downstream ordering is stable.
    """
    centroids = gdf.geometry.centroid
    inside = centroids.within(polygon)
    return gdf.loc[inside, id_column].tolist()


def run(
    cfg: dict[str, Any],
    scored: pd.DataFrame,
    treated_ids: list[str] | set[str],
    reduction_fraction: float,
) -> ScenarioResult:
    """Top-level entry point. ``cfg`` is accepted for future weight-driven
    variants but unused today (scenario math is closed-form on the scored frame)."""
    del cfg  # unused for now
    return apply_lighting_reduction(scored, treated_ids, reduction_fraction)
