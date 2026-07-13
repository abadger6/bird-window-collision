# Phase 1.75 — High-resolution ALAN (SDGSAT-1 Glimmer)

Design note for replacing/augmenting the 500 m VIIRS Black Marble lighting
term with **SDGSAT-1 Glimmer** imagery (10 m panchromatic / 40 m RGB). This
is the highest-value fix to the model's single weakest input — see
[methodology.md](methodology.md) limitation "ALAN is 500 m/pixel."

## Why

The `L` term enters the index multiplicatively
(`Risk_raw = Structure × norm(L) × class × edge`, see [risk.py](../pipeline/risk.py)).
At 500 m every building in a pixel inherits one radiance value, so an
isolated glass tower in an otherwise dark pixel gets `norm(L) ≈ 0` and its
structural hazard is annihilated. Two problems with one dataset fix:

1. **Spatial.** SDGSAT-1 Glimmer is 40 m multispectral / 10 m panchromatic —
   ~district-scale → ~block/building-cluster scale. Enough to tell a lit
   tower apart from its dark neighbor, which 500 m cannot.
2. **Spectral.** Glimmer has a **blue band (424–526 nm)**. Collision
   literature (Tan et al. 2023 Singapore MaxEnt; recent China ALAN studies)
   implicates blue / short-wavelength light as disproportionately attractive
   to nocturnal migrants. VIIRS DNB is panchromatic and *under*-senses blue.
   Glimmer lets us weight `L` by the collision-relevant spectrum.

## Approach — fusion, not replacement

Keep VIIRS Black Marble for what it is good at (daily, calibrated, seasonal
migration-month compositing — the existing `alan_composite` in
[config.yaml](../config.yaml)); use SDGSAT-1 for spatial + spectral detail.
Published VIIRS × SDGSAT-1 spatiotemporal fusion exists and is the model to
follow: temporal stability from VIIRS, per-block sharpness from Glimmer.

```
VIIRS Black Marble (500 m, daily)  ──► seasonal temporal signal  (have)
SDGSAT-1 Glimmer (40 m RGB, scene) ──► spatial + blue-band detail (new)
        └─────────► resample both to building centroids, blend ──► L_hires
```

Minimum viable version (recommended first cut): skip formal fusion. Sample
Glimmer radiance per building centroid as a **second** lighting column
(`alan_sdgsat`, plus `alan_sdgsat_blue`) alongside the existing VIIRS
`alan_radiance`. Run the risk model both ways and diff the top-N ranking. If
the leaderboard reorders meaningfully, that is direct evidence the 500 m
resolution was distorting results — a strong methodology-doc result and a
cheap experiment before committing to a fusion step.

## Data access & realities

- **Free** for research via the CBAS SDGSAT-1 Open Science Program
  (registration + data request). No ready-made Earth Engine ImageCollection
  (unlike Black Marble) — this is scene-based, so a fetch → mosaic →
  radiometric-calibrate → reproject step is required.
- **Sparse revisit.** Expect only a handful of usable cloud-free night
  scenes per city. Pick the best 1–3 over the migration-season window and
  mosaic; do not expect a dense time series.
- **Bands.** PAN 10 m (444–910 nm); RGB 40 m — blue 424–526, green
  505–612, red 600–894 nm. Use blue explicitly, not just luminance.
- **US coverage** exists (NYC / SF / Philadelphia appear in the literature);
  confirm Chicago scene availability before committing.

## Validation-only options (not pipeline inputs)

- **Jilin-1** — 0.92 m–30 cm *color* night imagery, genuinely per-building,
  but commercial/tasked (breaks the open, any-city model). Worth a one-time
  paid spot-check on marquee Chicago offenders to ground-truth the ranking.
- **ISS "Cities at Night"** — free color DSLR imagery, ~5–20 m effective,
  but oblique, uncalibrated, needs manual georeferencing. Qualitative
  cross-check only.
- **Luojia-1** — 130 m, decommissioned/archive, panchromatic. Skip; SDGSAT-1
  dominates it.

## Open decisions

Status as of the MVP experiment (pipeline/alan_hires.py + scripts/compare_lighting.py):

- **Config surface — resolved.** `assets.alan_hires` block landed in
  [config.yaml](../config.yaml) with per-city `sdgsat_scenes` list. Scenes are
  fetched manually into `data/sdgsat/<city>/` and mosaiced by
  [scripts/fetch_sdgsat.py](../scripts/fetch_sdgsat.py).
- **VIIRS-vs-Glimmer routing — resolved for MVP.** The risk model reads a
  single L column selected by `lighting_source` ∈ {`viirs`, `sdgsat_pan`,
  `sdgsat_blue`}. Where the chosen SDGSAT column is null (outside scene
  coverage), the model falls back to VIIRS per-row rather than dropping the
  building. Default is `viirs`, so existing runs are unchanged.
- **Blue band — deferred.** For MVP the blue band is a *substitute* L (its own
  `lighting_source` option), not a separate model term. If S5 shows
  `sdgsat_blue` reorders the top-N more than `sdgsat_pan` does, that's the
  signal to promote it to its own weighted term rather than blending.
- **Blend weighting — deferred to post-experiment.** `assets.alan_hires.blend`
  is a config stub (`mode: parallel`). Formal VIIRS × SDGSAT spatiotemporal
  fusion happens only if S5's leaderboard shows material reorder. The
  recommendation and thresholds are baked into
  [scripts/compare_lighting.py](../scripts/compare_lighting.py) `_auto_summary`.
- **Cloud/scene selection — still open.** Current mosaic step uses
  rasterio.merge's first-hit strategy; if `data/sdgsat/<city>/` holds a mix of
  clear and cloudy scenes, pre-mask before running or curate the directory to
  cloud-free scenes only. Documented as a TODO in `fetch_sdgsat.py`.
- **Chicago scene IDs — still open.** Placeholder `sdgsat_scenes: []` in
  config.yaml with search hints (Glimmer/GIU, cloud <10%, dev bbox, migration
  months 2022–2024). Populate once downloaded from the CBAS portal.

## References

- Weber et al. 2025. "Night lights from space: potential of SDGSAT-1 for
  ecological applications." *Remote Sensing in Ecology and Conservation.*
- VIIRS × SDGSAT-1 spatiotemporal fusion for daily observations, *Int. J.
  Digital Earth* 2025.
- Tan, D.J.X. et al. 2023. MaxEnt drivers of bird–building collisions,
  Singapore (bioRxiv 2023.06.27.546782) — blue-light collision link.
- SDGSAT-1 Open Science Program: https://www.sdgsat.ac.cn/
