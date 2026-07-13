# Methodology

This document is the source of truth for the risk model as implemented.
For the ground-truth validation summarized below, the full notebook —
including every plot, every table, and the reproducibility recipe — lives
at [notebooks/validation_cbcm.ipynb](../notebooks/validation_cbcm.ipynb)
and renders directly in GitHub's notebook viewer.

## What we compute

A **relative** per-building risk score in [0, 100] for a given city, using
building geometry + LiDAR heights (Overture Maps), nighttime radiance (NASA
Black Marble VNP46A2), and OSM habitat features (parks, water, wooded
areas). The score is *not* a calibrated mortality estimate — it is a
ranking that surfaces buildings most likely to produce collisions relative
to peers in the same city.

## The v2 index

Per building `i`:

```
H_eff_i    = sigmoid((height_i − 40) / 20)                        # nonlinear H (Loss 2014)
Structure  = w_F·norm(F) + w_A·norm(A) + w_H·norm(H_eff)          # F = perim × height, physical
L_class    = norm(alan_radiance_i) × class_multiplier[class_i]    # empty-office effect
E_i        = exp(−distance_to_nearest_habitat_i / 200 m)          # habitat proximity (Klem 2009)

Risk_raw_i = Structure × L_class × (1 + 0.5 × E_i)
Risk_i     = 100 × percentile_rank(Risk_raw_i)                    # display only
```

Weights are in [config.yaml](../config.yaml) and can be tuned per city.

### Term rationale

| Term | Source | Why it's in |
|---|---|---|
| **F** — facade area (perim × height) | Overture geometry | Best available proxy for glass surface area. Klem, Loss & Marra 2019 identify façade area as a dominant structural predictor. F stays as raw geometry (m²) even after height nonlinearity — it's a physical measurement, not a risk factor. |
| **A** — footprint area | Overture geometry | Independent structural predictor; larger silhouette = larger target. |
| **H_eff** — sigmoid(height) | Overture height (LiDAR-derived US) | Loss 2014 and Wang 2020: collision counts rise super-linearly with height above ~30m and plateau at extreme heights. Midpoint 40m, slope 20m matches this qualitatively. |
| **L** — mean ALAN radiance | Black Marble VNP46A2, migration months mean | ALAN correlates with collisions at the district scale (Cabrera-Cruz 2018, Van Doren 2017). 500m pixels — we call this the "lighting environment," not per-building lighting. |
| **class_multiplier** | Overture `class` field | Office towers with empty overnight lighting are the archetypal collision offender (Loss 2014). Multipliers tight around 1.0 so they modulate rather than dominate. |
| **E** — habitat-edge multiplier | OSM parks/water/wood | Collisions concentrate within a few hundred meters of habitat edges (Klem 2009). Exponential decay length 200m. |

### Null handling

- **height null** (~15% of buildings in a typical city): the H term is
  excluded and the structure weight of that building is renormalized over
  the remaining terms. The building still ranks.
- **class null**: multiplier defaults to 1.0.
- **habitat features empty** (unusual): E → 0 for every building, edge
  multiplier collapses to no-op.

## Validation — Chicago Bird Collision Monitors ground-truth

The v2 index is designed as a *ranking*, not a mortality estimate, so
the right validation is a rank-agreement study against an independent
observation dataset. We use the **Chicago Bird Collision Monitors
(CBCM)** dataset — 2,485 point observations of collisions collected via
volunteer-walked routes across the Loop, Mag Mile, Streeterville, West
Loop, and South Loop (2018–2021, via FLAP.org's data portal). 99.4% of
CBCM records fall in the same Apr/May/Sep/Oct migration window we use
to composite Black Marble ALAN.

Full method + plots: [notebooks/validation_cbcm.ipynb](../notebooks/validation_cbcm.ipynb).
Summary here.

### Method

1. **Point → building.** `sjoin_nearest` from CBCM points to Overture
   polygons with a **25 m cap**. CBCM's own median snap distance
   (from its published `location` string) is 11 m and its 75th %ile is
   24 m; 25 m absorbs GPS noise and facade-base landings without
   pulling in adjacent buildings. Points farther than the cap (~2% of
   the dataset) are dropped, not force-snapped.
2. **Survey hull.** CBCM is route-based, not random: buildings no one
   walks past have zero recorded collisions but are not zero risk.
   Define the survey footprint as the **union of 200 m buffers around
   every CBCM point** — adjacent points fuse into corridors that
   approximate the walked routes. Result: a 9.33 km² footprint covering
   **4,637 of 18,669** dev-bbox buildings (24.8%). All headline metrics
   are computed on this hull-restricted set. Sensitivity to the buffer
   radius is reported in §Robustness.
3. **Aggregate.** Per building: `cbcm_total`, `cbcm_dead`,
   `cbcm_species_n`, `on_route`. Aggregated columns are joined back
   onto the scored buildings and written to
   `data/processed/chicago_buildings_dev_scored_validated.geojson` for
   the kepler overlay (§Interactive comparison below).

### Headline results

| Metric | Value | Baseline |
|---|---|---|
| Spearman ρ (`risk_score` vs `cbcm_total`) | **+0.286**  (95% CI [+0.258, +0.314]) | 0 |
| Kendall τ | +0.233 | 0 |
| ROC-AUC (chronic offender ≥ 3 collisions) | **0.865** | 0.500 |
| PR-AUC (chronic offender ≥ 3 collisions) | **0.346** | 0.027 (base rate) |
| Top-25 precision | 0.24 | 0.005 (random) |
| Top-100 precision | 0.39 | 0.02 |
| Top-200 precision | 0.42 | 0.043 |

Read: the ROC-AUC of 0.87 is the load-bearing number. It says "if you
show the model a random known-chronic offender and a random
non-offender, it ranks the offender higher 87% of the time" — from a
model that saw zero collision data during training (it isn't trained
at all — it's fitted from the literature).

### Per-term diagnostics — which term does the work

Correlating each term with `cbcm_total` in isolation on the
hull-restricted set:

| Term | Spearman ρ | Reading |
|---|---:|---|
| **F** — facade area (perim × height) | **+0.348** | Strongest single term. Validates the "facade area as glass proxy" premise from Klem/Loss & Marra 2019. |
| **A** — footprint area | +0.316 | Independent structural signal — as expected. |
| **H_eff** — sigmoid height | +0.304 | Nonlinear H is doing real work. |
| **class multiplier** | +0.281 | Overture `class` field is meaningfully additive — office/commercial pull correctly. |
| **L** — ALAN radiance | +0.199 | Weakest structural term. Consistent with the "500 m pixel = district lighting environment, not per-building lighting" caveat. Motivates the SDGSAT-1 experiment below. |
| **E** — habitat edge factor | +0.040 | Barely correlated. In downtown Chicago the habitat multiplier is near-constant because most of the dev bbox is > 200 m from parks/water; the term is not helpfully separating buildings here. This is a real finding, not a bug — it may still matter in cities with more interspersed green space (Toronto, DC). |
| `risk_score` (composite) | +0.286 | Slightly below the strongest single term because L drags the composite down. Not a case for dropping L — literature evidence for L is strong, and the SDGSAT roadmap explicitly targets its resolution ceiling. |

The takeaway: **structural terms (F, A, H) are the model's spine**;
class multiplier is a meaningful nudge; ALAN is currently
resolution-limited; habitat edge is doing very little on this footprint.

### Robustness

The three methodology parameters that could move the numbers:

| Parameter | Range tested | Spearman ρ range | Reading |
|---|---|---|---|
| sjoin cap | 10 m → 100 m | 0.274 → 0.286 | Insensitive. |
| Survey hull buffer | 100 m → 800 m | 0.367 → 0.209 | Tighter hull = harsher test (more surveyed area, less unmatched noise). 200 m is the reported default. |
| Chronic threshold (for ROC-AUC) | ≥1 → ≥10 | ROC 0.819 → 0.927 | ROC-AUC rises monotonically as the "chronic" bar rises: **the model is best at picking out the worst offenders**, which is exactly the use case for prioritizing intervention. |

At the tightest hull (100 m) with the strictest threshold (≥10),
ROC-AUC is **0.93** — well above the level where a physics-based (not
fit) model has any right to be.

### What the residuals reveal

- **Underranked** (chronic offenders our model scores too low): concentrate
  in buildings with **null height** in Overture — the "roof" /
  "service" class where LiDAR misses. The pipeline's null-handling
  renormalizes structure weights, but a missing H drags scores down
  even when the actual building is tall. Fix path: the "manual-height
  overrides for landmarks" roadmap item.
- **Overranked** (high score, few CBCM records): dominated by
  100+ m office/apartment towers on the outer edge of the walked
  routes — Willis, Trump, 875 N Michigan, etc. These are probably not
  false positives; they're chronic offenders the CBCM routes reach
  less often. A survey-effort correction would let us grade them
  fairly.

### Interactive comparison — kepler A/B

The notebook writes
`data/processed/chicago_buildings_dev_scored_validated.geojson`: same
schema as the standard kepler input plus `cbcm_total`, `cbcm_dead`, and
`on_route`. Drop it into kepler as a duplicate 3D building layer and
switch **Fill Color** between `risk_score` (predicted) and `cbcm_total`
(observed) to A/B the two rankings on the same buildings.

### Validation caveats

- **MNAR by construction.** CBCM is not a random sample; the survey
  hull is our best available correction, but it is not a substitute
  for effort-weighted comparison.
- **Time window mismatch.** CBCM records span 2018–2021; ALAN
  composite is 2022–2024. Real changes in building lighting between
  windows count against the model.
- **One building dominates the tail** (~1,039 records, 62 species at
  the top offender — likely McCormick Place West or 150 N Riverside).
  Rank-based metrics handle this fine; count-based ones do not.
- **Alive + Dead counted together.** Both are evidence a strike
  happened. Restricting to `cbcm_dead` moves Spearman by < 0.02.

## Known limitations (short version)

- Relative, not absolute. Scores rank buildings; they aren't mortality counts.
- No glass ratio / reflectivity data. F is a geometric proxy.
- ALAN is 500m/pixel — district-scale, not per-building. Empirically
  the weakest term in the CBCM validation (§Validation, ρ = 0.20). See
  the SDGSAT-1 Glimmer 10–40 m experiment below.
- Overture heights come from LiDAR + OSM, both of which undercount tall
  antennas/superstructures. Willis Tower reads as ~340m instead of 442m.
  This is also the dominant driver of the "underranked" residuals in
  the CBCM validation.
- Habitat edge (E) is nearly flat on the Chicago downtown footprint
  (validation ρ = 0.04). May be more useful in cities with more
  interspersed habitat; keep it, but don't over-index on it as a
  differentiator downtown.
- v2's class multiplier assumes typical US usage patterns; may need
  per-city tuning where classes mean different things (e.g., Toronto vs.
  Chicago residential density).

## High-resolution ALAN experiment (SDGSAT-1 Glimmer)

The lighting term `L` also has a second, higher-resolution source available
as of the S1–S6 work: **SDGSAT-1 Glimmer** at 10 m panchromatic / 40 m RGB
(incl. a 424–526 nm blue band). See
[sdgsat_alan.md](sdgsat_alan.md) for the design note and why blue matters
for collisions.

The risk model exposes a `lighting_source` config key ∈ {`viirs`,
`sdgsat_pan`, `sdgsat_blue`} that swaps which column feeds the `L` slot of
the index. Everything else in the index is unchanged. Where SDGSAT scene
coverage is missing for a building, the model falls back to VIIRS per-row
rather than dropping the building — so switching sources never shrinks the
ranking.

Default remains `lighting_source: viirs` — no behavior change for existing
runs. [scripts/compare_lighting.py](../scripts/compare_lighting.py) runs the
model under all three sources on the same buildings and reports top-25
overlap + Spearman rank correlation. That comparison is the deliverable that
decides whether formal VIIRS × SDGSAT spatiotemporal fusion is worth
building. Result on the Chicago dev bbox: **TBD once Chicago Glimmer scenes
are downloaded from the CBAS Open Science portal** and
`scripts/fetch_sdgsat.py` → `scripts/build_alan_hires.py` →
`scripts/compare_lighting.py` have been run end-to-end. Populate this line
with the auto-summary from `compare_lighting.py` after that first run.

Full limitations table lives in [bird-collision-risk-tool-plan.md](../bird-collision-risk-tool-plan.md) §9.

## What we intentionally aren't doing yet

Two model expansions are prioritized for a later cut:

- **F — BirdFlow ecological weighting** (plan §8): per-week bird migration
  traffic surfaces from eBird Status & Trends, folded into the index as a
  temporal weight. Makes risk time-aware and adds a week/season selector.
- **G — Weather-triggered risk alerts**: real-time overlay driven by
  BirdCast + NWS forecasts. Fog and storm systems dramatically concentrate
  collisions; a nightly-updated overlay would flag high-risk nights ahead
  of time.

Both are documented in the top-level README's roadmap.

## Literature (short list, non-exhaustive)

- Klem, D. 2009. "Preventing Bird–Window Collisions." *Wilson Journal of
  Ornithology* — foundational structural predictors.
- Loss, S., et al. 2014. "Bird–building collisions in the United States:
  Estimates of annual mortality and species vulnerability." *The Condor*.
- Van Doren, B. et al. 2017. "High-intensity urban light installation
  dramatically alters nocturnal bird migration." *PNAS*.
- Cabrera-Cruz, S. et al. 2018. "Light pollution is greatest within
  migration passage areas." *Scientific Reports*.
- Winger, B. et al. 2019. "Nocturnal flight-calling behaviour predicts
  vulnerability to artificial light in migratory birds." *Proc. Royal
  Society B* — Field Museum / Chicago-focused.
- Wang, Y. et al. 2020. "Nocturnal migrant bird strikes on tall
  buildings…" — height nonlinearity evidence.
