"""T4 — unit tests for the risk-index math on tiny synthetic inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import risk


DEFAULT_WEIGHTS = {"facade": 0.5, "area": 0.2, "height": 0.3}


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Assemble a 4-column frame matching T2/T3 output shape."""
    df = pd.DataFrame(rows)
    for col in ("facade_area", "footprint_area", "height", "alan_radiance"):
        if col not in df.columns:
            df[col] = np.nan
    return df


# ----- min_max --------------------------------------------------------------


def test_min_max_basic():
    got = risk.min_max(pd.Series([0.0, 5.0, 10.0]))
    assert got.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_min_max_preserves_nulls():
    got = risk.min_max(pd.Series([0.0, np.nan, 10.0]))
    assert got.iloc[0] == 0.0
    assert np.isnan(got.iloc[1])
    assert got.iloc[2] == 1.0


def test_min_max_constant_series_returns_zeros():
    got = risk.min_max(pd.Series([7.0, 7.0, 7.0]))
    assert got.tolist() == [0.0, 0.0, 0.0]


def test_min_max_all_null_returns_all_null():
    got = risk.min_max(pd.Series([np.nan, np.nan]))
    assert got.isna().all()


# ----- weight validation ----------------------------------------------------


def test_weights_that_dont_sum_to_one_raise():
    df = _frame([{"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 1.0}])
    with pytest.raises(ValueError, match="sum to 1"):
        risk.compute(df, {"facade": 0.5, "area": 0.5, "height": 0.5})


# ----- structure_score with a full row --------------------------------------


def test_structure_score_matches_weighted_sum_when_all_terms_present():
    # Two-row frame — after min-max both rows normalize predictably.
    df = _frame([
        {"facade_area": 0.0, "footprint_area": 0.0, "height": 0.0, "alan_radiance": 0.0},
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 1.0},
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS)
    # Row 0: all norms are 0 -> structure_score = 0
    assert out.loc[0, "structure_score"] == pytest.approx(0.0)
    # Row 1: all norms are 1 -> structure_score = 0.5 + 0.2 + 0.3 = 1.0
    assert out.loc[1, "structure_score"] == pytest.approx(1.0)


# ----- structure_score with a partial row (null height) ---------------------


def test_structure_score_renormalizes_when_height_is_null():
    df = _frame([
        {"facade_area": 0.0, "footprint_area": 0.0, "height": 0.0,     "alan_radiance": 0.0},
        {"facade_area": 1.0, "footprint_area": 1.0, "height": np.nan,  "alan_radiance": 1.0},
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS)
    # Row 1 lost the H term. Weights present are facade (0.5) + area (0.2) = 0.7.
    # Both remaining norms are 1.0 -> structure = (0.5*1 + 0.2*1) / 0.7 = 1.0.
    # Ditto facade — with height null, facade_area = perim*height may or may
    # not be null; here we set it to 1.0 to isolate the height-null case.
    assert out.loc[1, "structure_score"] == pytest.approx(1.0)


def test_structure_score_returns_nan_when_all_terms_null():
    df = _frame([
        {"facade_area": 0.0, "footprint_area": 0.0, "height": 0.0, "alan_radiance": 0.5},
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 1.0},
        # Row 2 has no structural inputs at all.
        {"facade_area": np.nan, "footprint_area": np.nan, "height": np.nan, "alan_radiance": 0.7},
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS)
    assert np.isnan(out.loc[2, "structure_score"])
    assert np.isnan(out.loc[2, "risk_raw"])
    assert np.isnan(out.loc[2, "risk_score"])


# ----- risk_raw and risk_score end-to-end -----------------------------------


def test_risk_raw_is_structure_times_norm_alan():
    df = _frame([
        {"facade_area": 0.0, "footprint_area": 0.0, "height": 0.0, "alan_radiance": 0.0},
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 1.0},
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS)
    assert out.loc[0, "risk_raw"] == pytest.approx(0.0)  # 0 * 0
    assert out.loc[1, "risk_raw"] == pytest.approx(1.0)  # 1 * 1


def test_risk_score_is_percentile_rank_scaled_to_100():
    df = _frame([
        {"facade_area": v, "footprint_area": v, "height": v, "alan_radiance": v}
        for v in (0.0, 0.25, 0.5, 0.75, 1.0)
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS)
    # rank(pct=True) on 5 unique ascending values → 20, 40, 60, 80, 100.
    assert out["risk_score"].round(2).tolist() == [20.0, 40.0, 60.0, 80.0, 100.0]


def test_risk_is_monotone_in_lighting_holding_structure_fixed():
    # Baseline row breaks column constancy so min-max produces signal;
    # then we compare two identical-structure rows at different brightness.
    df = _frame([
        {"facade_area": 0.5, "footprint_area": 0.5, "height": 0.5, "alan_radiance": 5.0},
        {"facade_area": 2.0, "footprint_area": 2.0, "height": 2.0, "alan_radiance": 10.0},
        {"facade_area": 2.0, "footprint_area": 2.0, "height": 2.0, "alan_radiance": 100.0},
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS)
    assert out.loc[2, "risk_raw"] > out.loc[1, "risk_raw"]


def test_risk_is_monotone_in_structure_holding_lighting_fixed():
    # Baseline row breaks L column constancy; compare two identical-L rows
    # at different structural sizes.
    df = _frame([
        {"facade_area": 0.5, "footprint_area": 0.5, "height": 0.5, "alan_radiance": 50.0},
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 100.0},
        {"facade_area": 10.0, "footprint_area": 10.0, "height": 10.0, "alan_radiance": 100.0},
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS)
    assert out.loc[2, "risk_raw"] > out.loc[1, "risk_raw"]


# ----- input validation ------------------------------------------------------


def test_missing_input_column_raises_keyerror():
    df = pd.DataFrame([{"footprint_area": 1.0, "height": 1.0, "alan_radiance": 1.0}])
    with pytest.raises(KeyError, match="facade_area"):
        risk.compute(df, DEFAULT_WEIGHTS)


# ============================================================================
#  Model v2 — height response, class multipliers, edge factor
# ============================================================================


# ----- §C — sigmoid height response ----------------------------------------


def test_height_response_midpoint_returns_half():
    h = pd.Series([40.0, 40.0])
    out = risk.apply_height_response(h, midpoint_m=40, slope_m=20)
    assert out.tolist() == pytest.approx([0.5, 0.5])


def test_height_response_saturates_at_extremes():
    h = pd.Series([-1000.0, 1000.0])
    out = risk.apply_height_response(h, midpoint_m=40, slope_m=20)
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert out.iloc[1] == pytest.approx(1.0, abs=1e-6)


def test_height_response_preserves_nulls():
    h = pd.Series([10.0, np.nan, 100.0])
    out = risk.apply_height_response(h, midpoint_m=40, slope_m=20)
    assert not np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])
    assert not np.isnan(out.iloc[2])


def test_height_response_rejects_nonpositive_slope():
    with pytest.raises(ValueError):
        risk.apply_height_response(pd.Series([1.0]), midpoint_m=40, slope_m=0)


# ----- §B — class multiplier -----------------------------------------------


def test_class_multiplier_scales_risk_relative_to_baseline():
    df = _frame([
        {"id": "off", "class": "office", "facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 100.0},
        {"id": "res", "class": "residential", "facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 100.0},
        {"id": "unk", "class": "warehouse", "facade_area": 0.5, "footprint_area": 0.5, "height": 0.5, "alan_radiance": 50.0},
    ])
    multipliers = {"office": 1.3, "residential": 0.8}
    out = risk.compute(df, DEFAULT_WEIGHTS, class_multipliers=multipliers)
    # office vs residential — same structure and lighting, class multiplier ratio flows through.
    assert out.loc[0, "risk_raw"] > out.loc[1, "risk_raw"]
    assert out.loc[0, "class_multiplier"] == pytest.approx(1.3)
    assert out.loc[1, "class_multiplier"] == pytest.approx(0.8)
    # Unlisted class ('warehouse') defaults to 1.0.
    assert out.loc[2, "class_multiplier"] == pytest.approx(1.0)


def test_class_multiplier_column_missing_is_a_noop():
    df = _frame([
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 10.0},
        {"facade_area": 2.0, "footprint_area": 2.0, "height": 2.0, "alan_radiance": 20.0},
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS, class_multipliers={"office": 5.0})
    assert (out["class_multiplier"] == 1.0).all()


# ----- §A — habitat-edge multiplier ----------------------------------------


def test_edge_factor_applies_only_when_habitat_edge_present():
    df = _frame([
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 10.0, "habitat_edge": 1.0},
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 10.0, "habitat_edge": 0.0},
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS, edge_weight=0.5)
    # E=1 building gets Risk_raw × (1 + 0.5) = 1.5×; E=0 building unchanged.
    assert out.loc[0, "edge_factor"] == pytest.approx(1.5)
    assert out.loc[1, "edge_factor"] == pytest.approx(1.0)
    assert out.loc[0, "risk_raw"] == pytest.approx(1.5 * out.loc[1, "risk_raw"])


def test_edge_weight_zero_is_a_noop_even_with_habitat_column():
    df = _frame([
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 10.0, "habitat_edge": 1.0},
        {"facade_area": 2.0, "footprint_area": 2.0, "height": 2.0, "alan_radiance": 20.0, "habitat_edge": 0.0},
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS, edge_weight=0.0)
    assert (out["edge_factor"] == 1.0).all()


# ============================================================================
#  S4 — Selectable L source (VIIRS / SDGSAT pan / SDGSAT blue) + fallback
# ============================================================================


def _sdgsat_frame(rows: list[dict]) -> pd.DataFrame:
    """Same as _frame but also seeds the SDGSAT columns for L-source tests."""
    df = pd.DataFrame(rows)
    for col in (
        "facade_area", "footprint_area", "height",
        "alan_radiance", "alan_sdgsat", "alan_sdgsat_blue",
    ):
        if col not in df.columns:
            df[col] = np.nan
    return df


def test_default_lighting_source_is_viirs_and_matches_prior_behavior():
    """Backward-compat: computing without a source arg must equal computing
    with source='viirs'. The v2 index behavior is frozen for viirs."""
    df = _sdgsat_frame([
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 10.0, "alan_sdgsat": 999.0},
        {"facade_area": 2.0, "footprint_area": 2.0, "height": 2.0, "alan_radiance": 20.0, "alan_sdgsat": 999.0},
    ])
    default = risk.compute(df, DEFAULT_WEIGHTS)
    viirs = risk.compute(df, DEFAULT_WEIGHTS, lighting_source="viirs")
    assert default["risk_raw"].tolist() == pytest.approx(viirs["risk_raw"].tolist())
    # SDGSAT column shouldn't affect viirs scoring.
    df_nosdg = df.drop(columns=["alan_sdgsat", "alan_sdgsat_blue"])
    no_sdg = risk.compute(df_nosdg, DEFAULT_WEIGHTS)
    assert default["risk_raw"].tolist() == pytest.approx(no_sdg["risk_raw"].tolist())


def test_lighting_source_sdgsat_pan_uses_sdgsat_column():
    """When source is sdgsat_pan, norm(L) comes from alan_sdgsat, not VIIRS.
    Set VIIRS constant so a norm difference is unambiguously SDGSAT-driven."""
    df = _sdgsat_frame([
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 50.0, "alan_sdgsat": 0.0},
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 50.0, "alan_sdgsat": 100.0},
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS, lighting_source="sdgsat_pan")
    # SDGSAT normalizes 0→0, 100→1; risk_raw = structure * norm_L * mults.
    assert out.loc[0, "norm_L"] == pytest.approx(0.0)
    assert out.loc[1, "norm_L"] == pytest.approx(1.0)
    assert not out["L_fallback"].any()


def test_lighting_source_sdgsat_blue_uses_blue_column():
    df = _sdgsat_frame([
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0,
         "alan_radiance": 50.0, "alan_sdgsat": 0.0, "alan_sdgsat_blue": 0.0},
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0,
         "alan_radiance": 50.0, "alan_sdgsat": 0.0, "alan_sdgsat_blue": 100.0},
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS, lighting_source="sdgsat_blue")
    assert out.loc[0, "norm_L"] == pytest.approx(0.0)
    assert out.loc[1, "norm_L"] == pytest.approx(1.0)


def test_lighting_source_falls_back_to_viirs_on_null_sdgsat():
    """A building outside SDGSAT scene coverage (alan_sdgsat=NaN) must still
    get scored — its L comes from VIIRS instead of dropping to NaN."""
    df = _sdgsat_frame([
        {"id": "covered",   "facade_area": 1.0, "footprint_area": 1.0, "height": 1.0,
         "alan_radiance": 5.0,  "alan_sdgsat": 100.0},
        {"id": "uncovered", "facade_area": 1.0, "footprint_area": 1.0, "height": 1.0,
         "alan_radiance": 5.0,  "alan_sdgsat": np.nan},
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS, lighting_source="sdgsat_pan")
    assert out["L_fallback"].tolist() == [False, True]
    # No NaN in risk_raw despite the null SDGSAT input — fallback fixed it.
    assert out["risk_raw"].notna().all()
    # L_effective picks SDGSAT for row 0, VIIRS (5.0) for row 1.
    assert out.loc[0, "L_effective"] == pytest.approx(100.0)
    assert out.loc[1, "L_effective"] == pytest.approx(5.0)


def test_lighting_source_unknown_string_raises():
    df = _sdgsat_frame([
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 5.0},
    ])
    with pytest.raises(ValueError, match="lighting_source"):
        risk.compute(df, DEFAULT_WEIGHTS, lighting_source="jilin_1")


def test_lighting_source_sdgsat_missing_column_raises():
    """Requesting sdgsat_pan when alan_sdgsat wasn't built into the frame
    should error explicitly — not silently score everyone from VIIRS."""
    df = pd.DataFrame([
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 5.0},
    ])
    with pytest.raises(KeyError, match="alan_sdgsat"):
        risk.compute(df, DEFAULT_WEIGHTS, lighting_source="sdgsat_pan")


def test_norm_alan_radiance_still_reflects_viirs_under_sdgsat_source():
    """norm_alan_radiance stays a VIIRS-only sanity column so plots/inspection
    can compare it against norm_L regardless of the active lighting source."""
    df = _sdgsat_frame([
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 0.0,  "alan_sdgsat": 100.0},
        {"facade_area": 1.0, "footprint_area": 1.0, "height": 1.0, "alan_radiance": 10.0, "alan_sdgsat": 0.0},
    ])
    out = risk.compute(df, DEFAULT_WEIGHTS, lighting_source="sdgsat_pan")
    assert out["norm_alan_radiance"].tolist() == pytest.approx([0.0, 1.0])
    assert out["norm_L"].tolist() == pytest.approx([1.0, 0.0])  # inverted by SDGSAT
