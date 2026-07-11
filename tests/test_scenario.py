"""T5 — unit tests for the lighting-reduction scenario engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import risk, scenario


DEFAULT_WEIGHTS = {"facade": 0.5, "area": 0.2, "height": 0.3}


def _scored_frame(rows: list[dict]) -> pd.DataFrame:
    """Build a T4-shaped scored frame from tiny synthetic inputs."""
    df = pd.DataFrame(rows)
    for col in ("facade_area", "footprint_area", "height", "alan_radiance"):
        if col not in df.columns:
            df[col] = np.nan
    return risk.compute(df, DEFAULT_WEIGHTS)


# ----- input validation -----------------------------------------------------


def test_reduction_out_of_range_raises():
    df = _scored_frame([
        {"id": "a", "facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 10.0},
        {"id": "b", "facade_area": 2.0, "footprint_area": 2.0, "height": 2.0, "alan_radiance": 20.0},
    ])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        scenario.apply_lighting_reduction(df, ["a"], 1.5)
    with pytest.raises(ValueError):
        scenario.apply_lighting_reduction(df, ["a"], -0.1)


def test_missing_scored_column_raises():
    df = pd.DataFrame([{"id": "a", "alan_radiance": 1.0}])
    with pytest.raises(KeyError, match="structure_score"):
        scenario.apply_lighting_reduction(df, ["a"], 0.5)


# ----- boundary cases -------------------------------------------------------


def test_zero_reduction_is_a_noop():
    df = _scored_frame([
        {"id": "a", "facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 10.0},
        {"id": "b", "facade_area": 2.0, "footprint_area": 2.0, "height": 2.0, "alan_radiance": 100.0},
    ])
    result = scenario.apply_lighting_reduction(df, ["a", "b"], 0.0)
    assert result.risk_total_before == pytest.approx(result.risk_total_after)
    assert result.risk_removed == pytest.approx(0.0)


def test_full_reduction_zeroes_treated_risk():
    df = _scored_frame([
        {"id": "a", "facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 10.0},
        {"id": "b", "facade_area": 2.0, "footprint_area": 2.0, "height": 2.0, "alan_radiance": 100.0},
    ])
    # Frozen L range: L_min = 10 (row a's original). Treated L' = 0 -> clipped
    # to L_min effectively (post-clip norm = 0) — full treatment removes all
    # treated risk_raw.
    result = scenario.apply_lighting_reduction(df, ["a", "b"], 1.0)
    assert result.treated_deltas["risk_raw_after"].abs().sum() == pytest.approx(0.0)


# ----- the frozen-L-range guarantee -----------------------------------------


def test_untreated_buildings_are_exactly_unchanged():
    df = _scored_frame([
        {"id": "a", "facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 10.0},
        {"id": "b", "facade_area": 2.0, "footprint_area": 2.0, "height": 2.0, "alan_radiance": 50.0},
        {"id": "c", "facade_area": 3.0, "footprint_area": 3.0, "height": 3.0, "alan_radiance": 100.0},
    ])
    result = scenario.apply_lighting_reduction(df, ["c"], 0.5)

    # 'a' and 'b' were untreated. Their delta contribution must be zero.
    treated = set(result.treated_deltas["id"])
    assert treated == {"c"}
    # Also, the total risk removed equals c's delta alone.
    assert result.risk_removed == pytest.approx(result.treated_deltas["delta"].sum())


# ----- efficiency ranking ---------------------------------------------------


def test_treated_deltas_are_sorted_desc_by_delta():
    df = _scored_frame([
        {"id": "big_bright", "facade_area": 10.0, "footprint_area": 10.0, "height": 10.0, "alan_radiance": 100.0},
        {"id": "small_bright", "facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 100.0},
        {"id": "big_dim", "facade_area": 10.0, "footprint_area": 10.0, "height": 10.0, "alan_radiance": 10.0},
    ])
    result = scenario.apply_lighting_reduction(df, ["big_bright", "small_bright", "big_dim"], 0.5)

    ids_in_order = result.treated_deltas["id"].tolist()
    # Highest delta first. Bigger structure + brighter L should win.
    assert ids_in_order[0] == "big_bright"
    # Deltas monotonically non-increasing:
    deltas = result.treated_deltas["delta"].tolist()
    assert deltas == sorted(deltas, reverse=True)


# ----- ids_within_polygon --------------------------------------------------


def test_ids_within_polygon_matches_centroids():
    import geopandas as gpd
    from shapely.geometry import Point, box

    gdf = gpd.GeoDataFrame(
        {
            "id": ["inside", "outside"],
            "geometry": [Point(0, 0).buffer(0.5), Point(10, 10).buffer(0.5)],
        },
        crs="EPSG:26971",
    )
    poly = box(-1, -1, 1, 1)
    got = scenario.ids_within_polygon(gdf, poly)
    assert got == ["inside"]
