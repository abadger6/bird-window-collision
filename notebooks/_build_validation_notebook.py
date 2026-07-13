"""Builder for notebooks/validation_cbcm.ipynb.

Kept in-repo so the notebook can be regenerated after code changes without
hand-editing JSON. Run it via:

    uv run python notebooks/_build_validation_notebook.py

Then execute the notebook to embed outputs:

    uv run jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/validation_cbcm.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).parent / "validation_cbcm.ipynb"

nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (bird-collision-risk)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

cells: list = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source.strip("\n")))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source.strip("\n")))


# ─── 0. Header ──────────────────────────────────────────────────────────────
md(
    """
# Validation — Chicago Bird Collision Monitors (CBCM) ground-truth

**What this notebook does.** Takes the v2 risk index output for the Chicago
dev bbox and validates it against **2,485 CBCM observations** (Chicago Bird
Collision Monitors via FLAP, 2018–2021) covering the Loop, Mag Mile,
Streeterville, West Loop, and South Loop. Reports rank agreement, top-N
precision for the "known offenders" list, PR/ROC classification metrics for
chronic offenders, per-term diagnostics, and residual maps showing where
the model over- and under-ranks.

**Why this exists.** The tool produces a *relative* score. Without a
ground-truth comparison it is just a plausible ordering. This notebook is
the check that turns "plausible" into a numbers-backed domain claim.

**Links.** Model spec → [`docs/methodology.md`](../docs/methodology.md).
Pipeline README → [`README.md`](../README.md). Risk math source →
[`pipeline/risk.py`](../pipeline/risk.py).

**Reproduce.** Two prerequisite artifacts (both gitignored per the
project's `data/*` rule — regeneratable, not committed):

1. `data/processed/chicago_buildings_dev_scored.gpkg` — output of the
   five pipeline commands in the README (fetch → buildings → alan →
   risk → export).
2. `data/cities/Bird_Collision_Data_Map_view_chicago.geojson` — CBCM
   observation export, downloaded manually. Source dataset is public
   via the Chicago Bird Collision Monitors + FLAP.org data portal;
   drop the exported GeoJSON into `data/cities/` under the same name.

Then execute the notebook to embed outputs:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \\
    notebooks/validation_cbcm.ipynb
```

The final cell writes
`data/processed/chicago_buildings_dev_scored_validated.geojson` — same
schema as the kepler input plus three CBCM columns
(`cbcm_total`, `cbcm_dead`, `on_route`) so the same 3D building layer can
be recolored by observed collisions.
"""
)

# ─── 1. Setup ───────────────────────────────────────────────────────────────
md("## 1. Setup")

code(
    """
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from shapely.ops import unary_union
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="geopandas")

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "axes.titleweight": "semibold",
})

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
CBCM_PATH = REPO / "data/cities/Bird_Collision_Data_Map_view_chicago.geojson"
BLD_PATH  = REPO / "data/processed/chicago_buildings_dev_scored.gpkg"
PROJ_CRS  = "EPSG:26971"      # NAD83 / Illinois East — meters
WGS84     = "EPSG:4326"

# CBCM survey footprint = union of 200 m buffers around each point. Wider =
# more forgiving to the model; tighter = harsher. Sensitivity tested in §11.
HULL_BUFFER_M = 200
# sjoin_nearest cap. CBCM's own median snap distance is 11 m; 25 m allows
# for facade-base landings without pulling in adjacent buildings. See §11.
SNAP_MAX_M    = 25
# Chronic-offender positive threshold used in the binary classifier below.
CHRONIC_TOTAL = 3
"""
)

# ─── 2. Load data ───────────────────────────────────────────────────────────
md("## 2. Load data")

code(
    """
cbcm = gpd.read_file(CBCM_PATH).to_crs(PROJ_CRS)
bld  = gpd.read_file(BLD_PATH)      # already in EPSG:26971

# Parse CBCM's own "(distance from point: Xm)" tag — their per-record snap
# quality. Used later as a per-point precision histogram.
def parse_snap_m(loc: object) -> float:
    if not isinstance(loc, str):
        return np.nan
    m = re.search(r"distance from point:\\s*(\\d+)\\s*m", loc)
    return int(m.group(1)) if m else np.nan

cbcm["cbcm_snap_m"] = cbcm["location"].map(parse_snap_m)
cbcm["dateutc"]     = pd.to_datetime(cbcm["dateutc"], errors="coerce", utc=True)
cbcm["year"]        = cbcm["dateutc"].dt.year
cbcm["month"]       = cbcm["dateutc"].dt.month
cbcm["is_dead"]     = (cbcm["status"] == "Dead").astype(int)

print(f"CBCM points: {len(cbcm):,}   buildings scored: {len(bld):,}")
print(f"CBCM CRS: {cbcm.crs}   buildings CRS: {bld.crs}")
print(f"CBCM date range: {cbcm['dateutc'].min()}  →  {cbcm['dateutc'].max()}")
"""
)

# ─── 3. CBCM characterization ──────────────────────────────────────────────
md(
    """
## 3. CBCM observation data — characterization

Before comparing to the model we need to know what the ground-truth
actually looks like. The dataset is not random sampling: it is route-based
walked surveys by CBCM volunteers, so temporal, spatial, and effort biases
matter for how we read the final numbers.
"""
)

code(
    """
n_dead  = int(cbcm["is_dead"].sum())
n_alive = int((cbcm["status"] == "Alive").sum())
n_multi = int((cbcm["status"] == "Multiple").sum())
n_species = cbcm["species"].nunique()
print(f"total observations : {len(cbcm):,}")
print(f"  Dead             : {n_dead:,}  ({n_dead/len(cbcm)*100:.1f}%)")
print(f"  Alive            : {n_alive:,}  ({n_alive/len(cbcm)*100:.1f}%)")
print(f"  Multiple         : {n_multi:,}")
print(f"unique species     : {n_species}")
print(f"unique observers   : {cbcm['username'].nunique()}")
print(f"with 'location'    : {cbcm['location'].notna().sum():,}"
      f" ({cbcm['location'].notna().mean()*100:.0f}%)")
print(f"with parsed snap m : {cbcm['cbcm_snap_m'].notna().sum():,}")
print("CBCM own snap distance quantiles (m):",
      cbcm["cbcm_snap_m"].quantile([.5, .75, .9, .95]).round(1).to_dict())
"""
)

md("### 3.1 Temporal distribution")

code(
    """
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))

by_year = cbcm.groupby("year").size().reindex(range(int(cbcm["year"].min()),
                                                    int(cbcm["year"].max())+1),
                                                fill_value=0)
axes[0].bar(by_year.index.astype(int), by_year.values, color="#4a6fa5")
axes[0].set_title("CBCM observations by year")
axes[0].set_xlabel("year"); axes[0].set_ylabel("observations")

by_month = cbcm.groupby("month").size().reindex(range(1,13), fill_value=0)
migration = {4, 5, 9, 10}
colors = ["#c94f4f" if m in migration else "#a8b4c4" for m in by_month.index]
axes[1].bar(by_month.index, by_month.values, color=colors)
axes[1].set_title("CBCM observations by month (red = migration months)")
axes[1].set_xlabel("month"); axes[1].set_ylabel("observations")
axes[1].set_xticks(range(1,13))

plt.tight_layout(); plt.show()
"""
)

md("### 3.2 Species composition")

code(
    """
top_species = cbcm["species"].value_counts().head(20)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.barh(top_species.index[::-1], top_species.values[::-1], color="#4a6fa5")
ax.set_title(f"Top-20 species in CBCM Chicago dataset (of {n_species} total)")
ax.set_xlabel("observations")
plt.tight_layout(); plt.show()
"""
)

md("### 3.3 CBCM's own per-point snap precision")

code(
    """
snap = cbcm["cbcm_snap_m"].dropna()
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.hist(snap.clip(upper=200), bins=40, color="#4a6fa5", edgecolor="white")
ax.axvline(SNAP_MAX_M, color="#c94f4f", ls="--",
           label=f"our sjoin_nearest cap = {SNAP_MAX_M} m")
ax.set_xlabel("distance from CBCM point to building it was tagged with (m)")
ax.set_ylabel("observations")
ax.set_title("Per-observation snap distance already recorded by CBCM\\n"
             "(clipped at 200 m for the histogram)")
ax.legend()
plt.tight_layout(); plt.show()
print(f"share of records with snap <= {SNAP_MAX_M} m: "
      f"{(snap <= SNAP_MAX_M).mean()*100:.1f}%")
"""
)

md("### 3.4 Spatial density of raw CBCM points")

code(
    """
# Hexbin in projected meters — density directly interpretable as counts/hex.
fig, ax = plt.subplots(figsize=(8, 8))
hb = ax.hexbin(cbcm.geometry.x, cbcm.geometry.y, gridsize=40,
                cmap="magma", mincnt=1)
ax.set_aspect("equal")
ax.set_title("CBCM observation density (EPSG:26971)")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
cb = plt.colorbar(hb, ax=ax, label="observations per hex", shrink=0.7)
plt.tight_layout(); plt.show()
"""
)

# ─── 4. Point → building join ──────────────────────────────────────────────
md(
    """
## 4. Assign CBCM points to buildings

`sjoin_nearest` with a **25 m cap** — chosen because CBCM's own 75th-%ile
snap distance is 24 m and the median is 11 m. Anything farther is either
GPS noise on a wide street, a bird found in the middle of a plaza, or an
outright coding error — safest to drop.

When a point sits equidistant from two polygons (common at rowhouse
frontages), `sjoin_nearest` returns both. We dedupe by keeping the first
match per input row so per-building counts aren't inflated.
"""
)

code(
    """
joined = gpd.sjoin_nearest(
    cbcm, bld[["id", "geometry"]],
    how="left", max_distance=SNAP_MAX_M, distance_col="snap_m",
)
# Dedupe equidistant ties (see cell above)
joined = joined[~joined.index.duplicated(keep="first")].copy()

n_matched   = int(joined["id"].notna().sum())
n_unmatched = int(joined["id"].isna().sum())
print(f"CBCM points snapped to a building within {SNAP_MAX_M} m: "
      f"{n_matched:,} / {len(cbcm):,}  ({n_matched/len(cbcm)*100:.1f}%)")
print(f"unmatched (dropped from evaluation): {n_unmatched:,}")

fig, ax = plt.subplots(figsize=(8, 3.2))
ax.hist(joined["snap_m"].dropna(), bins=40, color="#4a6fa5", edgecolor="white")
ax.set_xlabel("snap distance CBCM point → nearest building (m)")
ax.set_ylabel("observations")
ax.set_title("Our sjoin_nearest snap distances")
plt.tight_layout(); plt.show()
"""
)

md("### 4.1 Aggregate to per-building CBCM counts")

code(
    """
cbcm_by_bld = (
    joined.dropna(subset=["id"])
          .groupby("id")
          .agg(cbcm_total=("id", "size"),
               cbcm_dead=("is_dead", "sum"),
               cbcm_species_n=("species", "nunique"),
               cbcm_first=("dateutc", "min"),
               cbcm_last=("dateutc", "max"))
          .reset_index()
)
print(f"buildings with ≥1 CBCM record: {len(cbcm_by_bld):,}")
print("distribution of cbcm_total across those buildings:")
print(cbcm_by_bld["cbcm_total"].describe().round(1).to_string())

merged = bld.merge(cbcm_by_bld, on="id", how="left")
for c in ("cbcm_total", "cbcm_dead", "cbcm_species_n"):
    merged[c] = merged[c].fillna(0).astype(int)
merged["cbcm_log1p"] = np.log1p(merged["cbcm_total"])
"""
)

md("### 4.2 Top-20 chronic offenders (observed)")

code(
    """
offenders = (
    merged.sort_values("cbcm_total", ascending=False)
          .head(20)
          [["id", "class", "height", "footprint_area",
            "risk_score", "cbcm_total", "cbcm_dead", "cbcm_species_n"]]
          .rename(columns={"footprint_area": "area_m2"})
)
offenders["height"]  = offenders["height"].round(1)
offenders["area_m2"] = offenders["area_m2"].round(0)
offenders["risk_score"] = offenders["risk_score"].round(1)
offenders.reset_index(drop=True)
"""
)

# ─── 5. Survey hull ────────────────────────────────────────────────────────
md(
    """
## 5. Define the evaluation footprint (survey hull)

**CBCM is a route-based survey.** Buildings that no one walks past can
have zero recorded collisions and still be lethal — the model should not
be penalized for "false positives" outside the walked routes. We handle
that by defining the evaluation footprint as the **union of 200 m buffers
around every CBCM point**. Adjacent points fuse into corridors that
approximate the walked routes; buildings outside the union are excluded
from headline metrics.

Sensitivity to the buffer radius is checked in §11.
"""
)

code(
    """
hull = unary_union(cbcm.geometry.buffer(HULL_BUFFER_M))
print(f"survey hull area:            {hull.area / 1e6:.2f} km²")
print(f"dev_bbox area (~35 km² per config)")

merged["on_route"] = merged.geometry.centroid.within(hull)
n_in = int(merged["on_route"].sum())
print(f"buildings inside hull:       {n_in:,} / {len(merged):,}"
      f"  ({n_in/len(merged)*100:.1f}%)")

eval_ = merged[merged["on_route"]].copy()
print(f"evaluation set (hull only):  {len(eval_):,}")
"""
)

md("### 5.1 Map — CBCM points, survey hull, dev bbox")

code(
    """
fig, ax = plt.subplots(figsize=(9, 9))

# All buildings in dev bbox — light grey
merged.plot(ax=ax, color="#e6e8ec", linewidth=0)
# Buildings on route — mid grey
eval_.plot(ax=ax, color="#c0c4cc", linewidth=0)
# Survey hull outline
gpd.GeoSeries([hull], crs=PROJ_CRS).boundary.plot(
    ax=ax, edgecolor="#c94f4f", linewidth=1.6, label="survey hull"
)
# CBCM points, colored by status
cbcm[cbcm["status"] == "Dead"].plot(ax=ax, color="#3b1f2b", markersize=4,
                                     alpha=0.6, label="Dead")
cbcm[cbcm["status"] == "Alive"].plot(ax=ax, color="#4a6fa5", markersize=4,
                                      alpha=0.6, label="Alive")

ax.set_aspect("equal")
ax.set_title("CBCM observations, survey hull, and buildings in scope")
ax.set_xlabel("x (m, EPSG:26971)"); ax.set_ylabel("y (m)")

handles = [
    mpatches.Patch(color="#e6e8ec", label="all dev-bbox buildings"),
    mpatches.Patch(color="#c0c4cc", label="on-route buildings (eval set)"),
    plt.Line2D([0], [0], color="#c94f4f", lw=1.6, label="survey hull"),
    plt.Line2D([0], [0], marker="o", ls="", color="#3b1f2b", label="Dead"),
    plt.Line2D([0], [0], marker="o", ls="", color="#4a6fa5", label="Alive"),
]
ax.legend(handles=handles, loc="lower right", frameon=True)
plt.tight_layout(); plt.show()
"""
)

# ─── 6. Rank agreement ────────────────────────────────────────────────────
md(
    """
## 6. Rank agreement — Spearman + Kendall

The v2 index is designed as a *ranking*, so rank-agreement statistics —
not RMSE or R² — are the right first cut. We compute both on the
hull-restricted evaluation set, and also on the whole dev bbox for
contrast (that number will be biased low by unsampled buildings).
"""
)

code(
    """
def rank_stats(df, label):
    x = df["risk_score"].values
    y = df["cbcm_total"].values
    rho, p_rho = spearmanr(x, y, nan_policy="omit")
    tau, p_tau = kendalltau(x, y, nan_policy="omit")
    print(f"{label:32s}  n={len(df):>5,}   "
          f"Spearman ρ = {rho:+.3f} (p={p_rho:.1e})   "
          f"Kendall τ = {tau:+.3f}")
    return rho, tau

rho_hull, tau_hull = rank_stats(eval_,  "hull-restricted (eval)")
rho_all,  tau_all  = rank_stats(merged, "all buildings (biased low)")

# Positive-only view — for the buildings that CBCM actually recorded, do
# higher scores correspond to more collisions? (Different question.)
pos = eval_[eval_["cbcm_total"] > 0]
rank_stats(pos, "positives only (cbcm ≥ 1)")
"""
)

md("### 6.1 Bootstrap 95% CI for Spearman ρ")

code(
    """
def bootstrap_spearman(x, y, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rhos = np.empty(n)
    for i in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        rhos[i], _ = spearmanr(x[s], y[s], nan_policy="omit")
    return rhos

rhos = bootstrap_spearman(eval_["risk_score"].values,
                           eval_["cbcm_total"].values)
lo, hi = np.percentile(rhos, [2.5, 97.5])
print(f"Spearman ρ (hull) = {rho_hull:+.3f}   95% CI = [{lo:+.3f}, {hi:+.3f}]")

fig, ax = plt.subplots(figsize=(7, 2.8))
ax.hist(rhos, bins=40, color="#4a6fa5", edgecolor="white")
ax.axvline(rho_hull, color="#c94f4f", lw=1.5, label=f"point estimate = {rho_hull:+.3f}")
ax.axvspan(lo, hi, color="#c94f4f", alpha=0.12, label=f"95% CI [{lo:+.3f}, {hi:+.3f}]")
ax.set_xlabel("bootstrap Spearman ρ"); ax.set_title("Bootstrap 95% CI (2000 resamples)")
ax.legend()
plt.tight_layout(); plt.show()
"""
)

md("### 6.2 Scatter — risk_score vs log(1 + cbcm_total)")

code(
    """
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

for ax, df, label in [(axes[0], eval_,  "hull-restricted"),
                       (axes[1], pos,    "positives only (cbcm ≥ 1)")]:
    ax.scatter(df["risk_score"], df["cbcm_log1p"],
                s=8, alpha=0.25, color="#4a6fa5", edgecolor="none")
    # Loess-lite: rolling mean over risk-percentile bins. Reindex over all
    # bins so bin centers stay aligned when some bins are empty.
    bins = np.linspace(0, 100, 21)
    labels = pd.IntervalIndex.from_breaks(bins, closed="right")
    df2 = df.assign(_b=pd.cut(df["risk_score"], bins, include_lowest=True))
    means = (df2.groupby("_b", observed=False)["cbcm_log1p"].mean()
                 .reindex(labels))
    centers = (bins[:-1] + bins[1:]) / 2
    m = means.notna().values
    ax.plot(centers[m], means.values[m], color="#c94f4f", lw=2, label="bin mean")
    r, _ = spearmanr(df["risk_score"], df["cbcm_total"], nan_policy="omit")
    ax.set_title(f"{label}   n={len(df):,}   ρ = {r:+.3f}")
    ax.set_xlabel("model risk_score (percentile)")
    ax.set_ylabel("log(1 + cbcm_total)")
    ax.legend(loc="upper left")

plt.tight_layout(); plt.show()
"""
)

# ─── 7. Top-N precision ────────────────────────────────────────────────────
md(
    """
## 7. Top-N precision — the "known offenders" question

Rank agreement is a whole-distribution number. The question a conservation
partner actually asks is: *"If we intervene on the top-N buildings our
model flags, how many are on CBCM's known-offender list?"* We compute
precision at N ∈ {10, 25, 50, 100, 200}.

We also compare against **random** as a baseline: with N/n_positive as the
expected precision if we ranked buildings at random.
"""
)

code(
    """
def top_n_precision(df, n, key_pred="risk_score", key_obs="cbcm_total"):
    pred = set(df.nlargest(n, key_pred)["id"])
    obs  = set(df[df[key_obs] > 0].nlargest(n, key_obs)["id"])
    return len(pred & obs) / n

n_pos = int((eval_["cbcm_total"] > 0).sum())
rows = []
for n in (10, 25, 50, 100, 200):
    p = top_n_precision(eval_, n)
    base = n_pos / len(eval_)          # random-guess precision
    rows.append({"N": n,
                 "precision": round(p, 3),
                 "random_baseline": round(base, 3),
                 "lift": round(p / base, 2) if base else None})
tab = pd.DataFrame(rows)
tab
"""
)

code(
    """
fig, ax = plt.subplots(figsize=(8, 3.6))
w = 0.35
ax.bar(np.arange(len(tab)) - w/2, tab["precision"], w,
        color="#4a6fa5", label="model")
ax.bar(np.arange(len(tab)) + w/2, tab["random_baseline"], w,
        color="#a8b4c4", label="random baseline")
ax.set_xticks(range(len(tab)))
ax.set_xticklabels([f"N={n}" for n in tab["N"]])
ax.set_ylabel("precision")
ax.set_title("Top-N precision — model vs. random baseline (hull-restricted)")
for i, (p, b) in enumerate(zip(tab["precision"], tab["random_baseline"])):
    ax.annotate(f"{p:.2f}", (i - w/2, p), ha="center", va="bottom", fontsize=9)
    ax.annotate(f"{b:.2f}", (i + w/2, b), ha="center", va="bottom", fontsize=9)
ax.legend()
plt.tight_layout(); plt.show()
"""
)

# ─── 8. PR / ROC classification ───────────────────────────────────────────
md(
    f"""
## 8. Binary classification — chronic-offender detection

Treat "chronic offender" as **cbcm_total ≥ 3** on the hull-restricted set,
and evaluate whether `risk_score` separates chronic from non-chronic. This
frames the model as a screening classifier.
"""
)

code(
    """
y_true  = (eval_["cbcm_total"] >= CHRONIC_TOTAL).astype(int).values
y_score = eval_["risk_score"].values

pr_auc  = average_precision_score(y_true, y_score)
roc_auc = roc_auc_score(y_true, y_score)
base    = y_true.mean()
print(f"positives (cbcm ≥ {CHRONIC_TOTAL}): {int(y_true.sum())} "
      f"/ {len(y_true)}  ({base*100:.2f}% of eval set)")
print(f"PR-AUC:  {pr_auc:.3f}   (random baseline = {base:.3f})")
print(f"ROC-AUC: {roc_auc:.3f}   (random baseline = 0.500)")

prec, rec, _ = precision_recall_curve(y_true, y_score)
fpr,  tpr, _ = roc_curve(y_true, y_score)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].plot(rec, prec, color="#4a6fa5", lw=2)
axes[0].axhline(base, color="#a8b4c4", ls="--",
                 label=f"random baseline = {base:.3f}")
axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision")
axes[0].set_title(f"Precision-Recall  (AP = {pr_auc:.3f})")
axes[0].legend()

axes[1].plot(fpr, tpr, color="#4a6fa5", lw=2)
axes[1].plot([0, 1], [0, 1], color="#a8b4c4", ls="--", label="random")
axes[1].set_xlabel("false-positive rate"); axes[1].set_ylabel("true-positive rate")
axes[1].set_title(f"ROC  (AUC = {roc_auc:.3f})")
axes[1].legend()

plt.tight_layout(); plt.show()
"""
)

md("### 8.1 Cumulative-gains curve")

code(
    """
order = np.argsort(-y_score)
y_sorted = y_true[order]
cum_pos = np.cumsum(y_sorted) / y_sorted.sum()
frac    = np.arange(1, len(y_sorted) + 1) / len(y_sorted)

fig, ax = plt.subplots(figsize=(6, 4.2))
ax.plot(frac, cum_pos, color="#4a6fa5", lw=2, label="model")
ax.plot([0, 1], [0, 1], color="#a8b4c4", ls="--", label="random")
ax.set_xlabel("fraction of buildings inspected (by risk_score, descending)")
ax.set_ylabel(f"fraction of chronic offenders (cbcm ≥ {CHRONIC_TOTAL}) caught")
ax.set_title("Cumulative-gains curve")

# Mark the 10% / 20% inspection budget lines
for f in (0.1, 0.2):
    y_at = np.interp(f, frac, cum_pos)
    ax.axvline(f, color="#c94f4f", ls=":", alpha=0.5)
    ax.annotate(f"inspect top {f*100:.0f}% → catch {y_at*100:.0f}%",
                (f, y_at), textcoords="offset points", xytext=(6, -14),
                fontsize=9, color="#c94f4f")
ax.legend(loc="lower right")
plt.tight_layout(); plt.show()
"""
)

# ─── 9. Calibration ───────────────────────────────────────────────────────
md(
    """
## 9. Calibration — does higher score = more collisions?

The model output is a percentile, not a probability, so classical
calibration curves don't strictly apply. But the analogous check is: bin
buildings by risk-score decile on the hull-restricted set and check that
**mean observed collisions rises monotonically** with score decile.
"""
)

code(
    """
eval_["risk_decile"] = pd.qcut(eval_["risk_score"], 10,
                                 labels=range(1, 11), duplicates="drop")
cal = (eval_.groupby("risk_decile", observed=True)
             .agg(n_bld=("id", "size"),
                  mean_cbcm=("cbcm_total", "mean"),
                  frac_with_hit=("cbcm_total", lambda s: (s > 0).mean()),
                  frac_chronic=("cbcm_total", lambda s: (s >= CHRONIC_TOTAL).mean()))
             .reset_index())

fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))

axes[0].bar(cal["risk_decile"].astype(int), cal["mean_cbcm"],
             color="#4a6fa5")
axes[0].set_xlabel("risk_score decile (1 = lowest, 10 = highest)")
axes[0].set_ylabel("mean CBCM collisions per building")
axes[0].set_title("Mean observed collisions by predicted risk decile")

axes[1].plot(cal["risk_decile"].astype(int), cal["frac_with_hit"],
              marker="o", color="#4a6fa5", label="≥ 1 CBCM record")
axes[1].plot(cal["risk_decile"].astype(int), cal["frac_chronic"],
              marker="s", color="#c94f4f",
              label=f"chronic (≥ {CHRONIC_TOTAL})")
axes[1].set_xlabel("risk_score decile")
axes[1].set_ylabel("fraction of buildings")
axes[1].set_title("Hit-rate by predicted risk decile")
axes[1].legend()

plt.tight_layout(); plt.show()
cal
"""
)

# ─── 10. Term-level analysis ──────────────────────────────────────────────
md(
    """
## 10. Which term does the work?

The v2 index is a product of five terms — facade area (F), footprint (A),
sigmoid-height (H_eff), ALAN (L), class multiplier, and habitat edge.
Correlating each term with observed collisions in isolation tells us
which are pulling their weight in this dataset.
"""
)

code(
    """
term_cols = [
    ("norm_facade_area",       "facade area (F)"),
    ("norm_footprint_area",    "footprint area (A)"),
    ("norm_height_effective",  "sigmoid height (H_eff)"),
    ("norm_alan_radiance",     "ALAN radiance (L)"),
    ("class_multiplier",       "class multiplier"),
    ("edge_factor",            "habitat edge factor"),
    ("structure_score",        "structure_score (composite)"),
    ("risk_raw",               "risk_raw (final product)"),
    ("risk_score",             "risk_score (percentile)"),
]
rows = []
for col, label in term_cols:
    r, p = spearmanr(eval_[col], eval_["cbcm_total"], nan_policy="omit")
    rows.append({"term": label, "spearman_rho": round(r, 3), "p": p})
term_tbl = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(8, 4.2))
colors = ["#4a6fa5" if r >= 0 else "#c94f4f" for r in term_tbl["spearman_rho"]]
ax.barh(term_tbl["term"][::-1], term_tbl["spearman_rho"][::-1], color=colors[::-1])
ax.axvline(0, color="#333", lw=0.5)
ax.set_xlabel("Spearman ρ vs. cbcm_total (hull-restricted)")
ax.set_title("Per-term rank agreement with observed collisions")
for i, r in enumerate(term_tbl["spearman_rho"][::-1]):
    ax.annotate(f"{r:+.2f}", (r, i), xytext=(4 if r >= 0 else -4, 0),
                textcoords="offset points",
                ha="left" if r >= 0 else "right", va="center", fontsize=9)
plt.tight_layout(); plt.show()
term_tbl
"""
)

# ─── 11. Residuals ────────────────────────────────────────────────────────
md(
    """
## 11. Residuals — where the model misses

**Underranked** = high CBCM count, low risk_score. These are the model's
false negatives — buildings that are demonstrably lethal but our physics-
based index doesn't flag. This is where we most need model improvements
(e.g., per-building glass ratio, blue-band ALAN).

**Overranked** = high risk_score, low CBCM count. These are trickier —
they can be either legitimate false positives OR simply buildings the
walked routes never reach. We show both.
"""
)

code(
    """
# Residual = normalized cbcm count minus normalized score.
def rank_pct(s):
    return s.rank(pct=True) * 100

eval_["cbcm_rank_pct"] = rank_pct(eval_["cbcm_total"])
eval_["residual"]      = eval_["cbcm_rank_pct"] - eval_["risk_score"]

under = (eval_[eval_["cbcm_total"] >= CHRONIC_TOTAL]
          .sort_values("residual", ascending=False)
          .head(15)
          [["id", "class", "height", "footprint_area",
            "risk_score", "cbcm_total", "cbcm_dead", "residual"]])
under["height"] = under["height"].round(1)
under["footprint_area"] = under["footprint_area"].round(0)
print("Top-15 underranked buildings (chronic offenders our model misses):")
under.reset_index(drop=True)
"""
)

code(
    """
over = (eval_.sort_values("residual", ascending=True)
              .head(15)
              [["id", "class", "height", "footprint_area",
                "risk_score", "cbcm_total", "cbcm_dead", "residual"]])
over["height"] = over["height"].round(1)
over["footprint_area"] = over["footprint_area"].round(0)
print("Top-15 overranked buildings (high score, few CBCM records — may be "
      "genuine false positives OR unsurveyed):")
over.reset_index(drop=True)
"""
)

md("### 11.1 Residual map")

code(
    """
fig, ax = plt.subplots(figsize=(10, 10))
eval_.plot(ax=ax, color="#e6e8ec", linewidth=0)

# Underranked (missed by model) — red highlight
under_ids = set(under["id"])
u = eval_[eval_["id"].isin(under_ids)]
u.plot(ax=ax, color="#c94f4f", edgecolor="black", linewidth=0.4,
        label="underranked (model miss)")

# Overranked
over_ids = set(over["id"])
o = eval_[eval_["id"].isin(over_ids)]
o.plot(ax=ax, color="#f2c14e", edgecolor="black", linewidth=0.4,
        label="overranked")

gpd.GeoSeries([hull], crs=PROJ_CRS).boundary.plot(
    ax=ax, edgecolor="#4a6fa5", linewidth=1.2, ls="--"
)

ax.set_aspect("equal")
ax.set_title("Residual map — top model misses (red) and possible false "
              "positives (yellow)")
handles = [
    mpatches.Patch(color="#c94f4f", label="underranked (chronic offender, low score)"),
    mpatches.Patch(color="#f2c14e", label="overranked (high score, low CBCM)"),
    plt.Line2D([0], [0], color="#4a6fa5", lw=1.2, ls="--", label="survey hull"),
]
ax.legend(handles=handles, loc="lower right")
plt.tight_layout(); plt.show()
"""
)

# ─── 12. Robustness ───────────────────────────────────────────────────────
md(
    """
## 12. Robustness — how much do methodology choices move the numbers?

Three parameter sweeps: the sjoin_nearest snap cap, the chronic-offender
positive threshold, and the survey-hull buffer radius. If headline
Spearman shifts more than ±0.05 across the plausible range, the finding
is method-dependent and we should say so.
"""
)

code(
    """
def eval_with(snap_max, hull_buffer):
    j = gpd.sjoin_nearest(cbcm, bld[["id", "geometry"]], how="left",
                           max_distance=snap_max, distance_col="_d")
    j = j[~j.index.duplicated(keep="first")]
    agg = (j.dropna(subset=["id"]).groupby("id").size()
            .rename("cbcm_total").reset_index())
    m = bld.merge(agg, on="id", how="left").fillna({"cbcm_total": 0})
    h = unary_union(cbcm.geometry.buffer(hull_buffer))
    m["on_route"] = m.geometry.centroid.within(h)
    e = m[m["on_route"]]
    r, _ = spearmanr(e["risk_score"], e["cbcm_total"], nan_policy="omit")
    return r, len(e), int((e["cbcm_total"] > 0).sum())

# Sweep 1: snap threshold
sweep_snap = pd.DataFrame(
    [{"snap_max_m": s,
      "spearman": round(eval_with(s, HULL_BUFFER_M)[0], 3),
      "n_eval":   eval_with(s, HULL_BUFFER_M)[1],
      "n_positive": eval_with(s, HULL_BUFFER_M)[2]}
     for s in (10, 25, 50, 100)]
)
print("Snap-threshold sweep (hull buffer = 200 m fixed):")
print(sweep_snap.to_string(index=False))
"""
)

code(
    """
# Sweep 2: hull buffer
sweep_hull = pd.DataFrame(
    [{"hull_buffer_m": b,
      "spearman": round(eval_with(SNAP_MAX_M, b)[0], 3),
      "n_eval":   eval_with(SNAP_MAX_M, b)[1],
      "n_positive": eval_with(SNAP_MAX_M, b)[2]}
     for b in (100, 200, 400, 800)]
)
print("Hull-buffer sweep (snap cap = 25 m fixed):")
print(sweep_hull.to_string(index=False))

# Sweep 3: chronic threshold's effect on PR-AUC
rows = []
for k in (1, 2, 3, 5, 10):
    y = (eval_["cbcm_total"] >= k).astype(int).values
    if y.sum() < 5:
        continue
    rows.append({
        "chronic_threshold": k,
        "n_positive": int(y.sum()),
        "PR_AUC":  round(average_precision_score(y, eval_["risk_score"]), 3),
        "ROC_AUC": round(roc_auc_score(y, eval_["risk_score"]), 3),
    })
sweep_thr = pd.DataFrame(rows)
print("\\nChronic-threshold sweep (snap = 25 m, hull = 200 m):")
print(sweep_thr.to_string(index=False))
"""
)

# ─── 13. Cuts ─────────────────────────────────────────────────────────────
md(
    """
## 13. Time & species cuts

Two focused checks: does model agreement improve when we restrict CBCM to
migration months (matching the ALAN composite window), and does it differ
by species life-history? The latter would suggest a per-species weighting
term (roadmap "species-specific weighting").
"""
)

code(
    """
mig = cbcm[cbcm["month"].isin({4, 5, 9, 10})]
print(f"CBCM restricted to Apr/May/Sep/Oct: {len(mig):,} of {len(cbcm):,}"
      f"  ({len(mig)/len(cbcm)*100:.1f}%)")

# Rebuild per-building aggregates for the migration-only subset
def rebuild_agg(points):
    j = gpd.sjoin_nearest(points, bld[["id", "geometry"]], how="left",
                           max_distance=SNAP_MAX_M, distance_col="_d")
    j = j[~j.index.duplicated(keep="first")]
    return (j.dropna(subset=["id"]).groupby("id").size()
             .rename("cbcm_total").reset_index())

agg_mig = rebuild_agg(mig)
m2 = bld.merge(agg_mig, on="id", how="left").fillna({"cbcm_total": 0})
m2["on_route"] = m2.geometry.centroid.within(hull)
e2 = m2[m2["on_route"]]
r_mig, _ = spearmanr(e2["risk_score"], e2["cbcm_total"], nan_policy="omit")
print(f"Spearman ρ (migration-months only, hull-restricted): {r_mig:+.3f}"
      f"   vs. all-months {rho_hull:+.3f}")
"""
)

code(
    """
# Per-species rho for the top species with enough records
per_sp = []
for sp, cnt in cbcm["species"].value_counts().items():
    if cnt < 40:
        break
    sub = cbcm[cbcm["species"] == sp]
    agg_s = rebuild_agg(sub)
    ms = bld.merge(agg_s, on="id", how="left").fillna({"cbcm_total": 0})
    ms["on_route"] = ms.geometry.centroid.within(hull)
    es = ms[ms["on_route"]]
    if es["cbcm_total"].sum() < 10:
        continue
    r, _ = spearmanr(es["risk_score"], es["cbcm_total"], nan_policy="omit")
    per_sp.append({"species": sp, "n_obs": cnt, "spearman": round(r, 3)})
per_sp_tbl = pd.DataFrame(per_sp)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(per_sp_tbl["species"][::-1], per_sp_tbl["spearman"][::-1],
         color=["#4a6fa5" if r >= 0 else "#c94f4f"
                for r in per_sp_tbl["spearman"][::-1]])
ax.axvline(rho_hull, color="#333", ls="--", lw=1,
            label=f"pooled ρ = {rho_hull:+.3f}")
ax.set_xlabel("per-species Spearman ρ (risk_score vs cbcm_total)")
ax.set_title("Model agreement by CBCM species (≥40 records, "
              "≥10 hull hits)")
ax.legend()
plt.tight_layout(); plt.show()
per_sp_tbl
"""
)

# ─── 14. Export ──────────────────────────────────────────────────────────
md(
    """
## 14. Export augmented GeoJSON for kepler overlay

Same schema as `data/processed/chicago_buildings_dev_scored.geojson`, plus
three columns:

- `cbcm_total`  — total CBCM records snapped to this building
- `cbcm_dead`   — dead-only subset
- `on_route`    — whether the building sits inside the survey hull

In kepler, add this file as a second copy of the same 3D extrusion layer
and switch **Fill Color** between `risk_score` (predicted) and
`cbcm_total` (observed) to A/B the two rankings on the same buildings.
"""
)

code(
    """
out_cols = list(bld.columns) + ["cbcm_total", "cbcm_dead", "on_route"]
export = merged[out_cols].copy()
export = export.to_crs(WGS84)     # kepler expects WGS84

OUT_PATH = REPO / "data/processed/chicago_buildings_dev_scored_validated.geojson"
export.to_file(OUT_PATH, driver="GeoJSON")
print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size / 1e6:.1f} MB, "
      f"{len(export):,} features)")
"""
)

# ─── 15. Caveats ─────────────────────────────────────────────────────────
md(
    f"""
## 15. Caveats

**Read the headline numbers with these in mind:**

1. **CBCM is route-based, not random.** Buildings outside the walked
   corridors have zero recorded collisions and still may be lethal. We
   handle this with the survey hull (§5), but every hull rule is a
   judgment call.
2. **2018–2021 window.** The model's ALAN input composites 2022–2024
   Black Marble. If a chronic offender changed lighting behavior between
   the CBCM window and now, we would penalize the model for a real
   change in the world.
3. **One building dominates the tail** (~1,000 records at the top
   offender). Rank-based metrics (Spearman) handle this fine; count-based
   ones don't. That's why we lead with Spearman + top-N precision, not
   MAE.
4. **CBCM records living-bird strikes as well as fatalities.** Both are
   direct evidence a strike happened, so we count both. Filter to
   `cbcm_dead` if you want the strict mortality view — Spearman moves
   very little.
5. **Class multiplier is one of the strongest single terms** here — but
   that's an in-sample observation. If we tuned the multipliers to CBCM
   we would overfit; the current values come from Loss 2014, unchanged.

**Where the residuals point us next** (from §11):

- Underranked buildings tend to be tall, high-glass structures the model
  already scores well but not top-decile. Adding a **glass ratio** term
  (roadmap "Glass ratio from Street View / satellite imagery ML") is
  what would close that gap.
- Overranked buildings concentrate in low-CBCM-coverage areas — probably
  survey artifacts, not model errors. A finer survey-effort correction
  would let us evaluate them fairly.
"""
)

# ─── Write ───────────────────────────────────────────────────────────────
nb["cells"] = cells
nbf.write(nb, OUT)
print(f"wrote {OUT}  ({len(cells)} cells)")
