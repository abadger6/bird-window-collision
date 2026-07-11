# Methodology

This document is the source of truth for the risk model as implemented.
For strategy and roadmap, see
[bird-collision-risk-tool-plan.md](../bird-collision-risk-tool-plan.md).

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

## Known limitations (short version)

- Relative, not absolute. Scores rank buildings; they aren't mortality counts.
- No glass ratio / reflectivity data. F is a geometric proxy.
- ALAN is 500m/pixel — district-scale, not per-building.
- Overture heights come from LiDAR + OSM, both of which undercount tall
  antennas/superstructures. Willis Tower reads as ~340m instead of 442m.
- v2's class multiplier assumes typical US usage patterns; may need
  per-city tuning where classes mean different things (e.g., Toronto vs.
  Chicago residential density).

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
