# Bird–Window Collision Risk Tool

Relative bird–window collision risk score for every building in a city,
driven by **building geometry + LiDAR heights** (Overture Maps),
**nighttime radiance** (NASA Black Marble), and **habitat proximity**
(OpenStreetMap). Rendered as an interactive 3D map via
[kepler.gl](https://kepler.gl). Chicago is the pilot city; the whole
pipeline re-runs for any other municipality by adding a block under
`cities:` in `config.yaml` and switching `active_city`.

Model math + literature citations: [docs/methodology.md](docs/methodology.md).
Ground-truth validation vs. CBCM: [notebooks/validation_cbcm.ipynb](notebooks/validation_cbcm.ipynb)
(Spearman ρ = 0.29, ROC-AUC = 0.87 on 4,637 downtown buildings against
2,485 Chicago Bird Collision Monitors observations).
Strategy + roadmap: [bird-collision-risk-tool-plan.md](bird-collision-risk-tool-plan.md).

## Status

Phase 1 pilot on Chicago downtown (Loop + Mag Mile + Streeterville + West
Loop + South Loop through McCormick Place — ~35 km², ~18k buildings). Risk
model is at v2: habitat-edge multiplier, per-class L weighting, and
nonlinear H response.

## Setup (one-time)

```bash
uv sync                                # create .venv + install deps
uv run earthengine authenticate        # one-time browser flow
# then set gee.project in config.yaml to your Cloud project id
uv run python scripts/check_gee_auth.py
```

**Why GEE:** we use Earth Engine only to serve up the Black Marble VIIRS
nighttime-lights ImageCollection and sample radiance per building
centroid. Nothing gets published back to GEE — outputs stay local, and
the public visualization lives in kepler.gl. You still need a
Cloud project registered for Earth Engine ([console.cloud.google.com/earth-engine](https://console.cloud.google.com/earth-engine)).

### Intel Mac note

`cryptography` (pulled in transitively via `earthengine-api` →
`google-auth`) stopped shipping x86_64 macOS wheels around v45, so on an
Intel Mac it builds from source. First-time setup:

```bash
brew install pkgconf openssl@3
export OPENSSL_DIR="$(brew --prefix openssl@3)"
export PKG_CONFIG_PATH="$(brew --prefix openssl@3)/lib/pkgconfig"
uv sync
```

Apple Silicon and Linux users hit prebuilt wheels — no extra steps.

## Pipeline — the five commands

Each script reads config.yaml and writes an artifact the next script
consumes. Run in order. Dev bbox (~35 km²) takes ~4 min end-to-end;
full-city Chicago will take ~30 min.

```bash
# 1. Fetch building footprints for the active city's dev bbox from Overture S3
uv run python scripts/fetch_overture_city.py
# → data/cities/chicago_dev.geojson  (~15–20k polygons)

# 2. Clean geometry, backfill heights, compute footprint_area/perimeter/facade_area
uv run python scripts/build_buildings.py
# → data/processed/chicago_buildings_dev.gpkg

# 3. Sample Black Marble nighttime radiance (mean over migration months) per centroid
uv run python scripts/build_alan.py
# → data/processed/chicago_buildings_dev_alan.gpkg

# 4. Compute v2 risk index (pulls OSM habitat, applies class + edge multipliers)
uv run python scripts/build_risk.py
# → data/processed/chicago_buildings_dev_scored.gpkg

# 5. Export a kepler.gl-ready GeoJSON
uv run python scripts/export_geojson.py
# → data/processed/chicago_buildings_dev_scored.geojson
```

Add `--full` to steps 1–5 for the full-city bbox instead of the dev slice
(warning: ALAN sampling in step 3 will take much longer at 500k+
centroids).

### Two useful side scripts

```bash
# Ranked-list demo of the T5 scenario engine (top-N lighting reduction)
uv run python scripts/run_scenario.py --top 25 --reduction 0.5

# Print schema + coverage of a fetched Overture GeoJSON
uv run python scripts/inspect_local_footprints.py
```

## Publishing to kepler.gl (public POC path)

1. Open [kepler.gl/demo](https://kepler.gl/demo), drag
   `data/processed/chicago_buildings_dev_scored.geojson` onto the drop zone.
2. In the Layers panel: set **Fill Color** based on `risk_score` (YlOrRd
   palette), toggle **3D Buildings** on with **Height** = `height`.
3. Tune camera to a nice angle. **Export Map → HTML**, save as `index.html`.
4. Drop that `index.html` into `docs/`, commit, push. GitHub Pages redeploys
   at `https://<user>.github.io/<repo>/` within ~1 min.

See [docs/README.md](docs/README.md) for the update workflow.

## Data & licensing

- **Buildings:** [Overture Maps Foundation](https://overturemaps.org/) via
  the official `overturemaps` Python client (streams from
  `s3://overturemaps-us-west-2`). **ODbL** — attribution required on any
  redistribution (including the public app).
- **ALAN:** NASA Black Marble daily (`NASA/VIIRS/002/VNP46A2`), 500 m/pixel.
  Treat as a district-scale lighting-environment proxy, not per-building
  lighting.
- **Habitat features:** OpenStreetMap (parks, water, wooded areas) via
  `osmnx`. **ODbL**, attribution required.

## Roadmap — near-term

Ordered by cost/value; see [docs/methodology.md](docs/methodology.md) for
the model-side rationale.

- **Full-city scale-up (Chicago).** Once ALAN batching is stress-tested at
  500k centroids, run the pipeline against the full city bbox and republish.
- **Validation vs. CBCM data.** ✅ First pass done —
  [notebooks/validation_cbcm.ipynb](notebooks/validation_cbcm.ipynb). On
  2,485 CBCM observations restricted to the survey hull (4,637
  buildings): Spearman ρ = 0.29, top-25 precision = 0.24 (vs. 0.005
  random baseline, 46× lift), PR-AUC = 0.35 for chronic-offender
  detection. Notebook also writes
  `data/processed/chicago_buildings_dev_scored_validated.geojson` for a
  predicted-vs-observed A/B in kepler. Next: FLAP Canada (Toronto) and
  NYC Bird Alliance data for a multi-city cohort.
- **Manual-height overrides for landmarks.** Add per-city `landmarks`
  block in config.yaml to hard-code height for buildings where Overture
  LiDAR under-reports (Willis Tower antenna, McCormick Place footprint
  without height).
- **High-res ALAN (SDGSAT-1 Glimmer).** Replace/augment the 500 m VIIRS
  lighting term with 40 m RGB / 10 m panchromatic SDGSAT-1 Glimmer — fixes
  the model's weakest input (resolution) and adds a blue band (the
  collision-relevant spectrum) in one dataset. Start with a per-building
  side-by-side sampling experiment, not full fusion. Design:
  [docs/sdgsat_alan.md](docs/sdgsat_alan.md).
- **Multi-city cohort.** Add NYC (NYC Bird Alliance data) and Toronto
  (FLAP Canada data). Multi-city moves us off the kepler HTML-export path
  and into an embedded frontend.

## Roadmap — Phase 2

- **F — BirdFlow ecological weighting** (plan §8): per-week bird
  migration-traffic surfaces from eBird Status & Trends, folded into the
  index as a temporal weight `M(week)`. Makes risk time-aware, adds a
  week/season selector. Runs offline via `BirdFlowPy`; ingested as GEE
  ImageCollection.
- **G — Weather-triggered risk alerts.** Real-time overlay driven by
  BirdCast migration-intensity forecasts + NWS fog/storm forecasts. Fog
  and low cloud dramatically concentrate collisions; a nightly-updated
  overlay would flag high-risk nights ahead of time. Deploy on Modal or
  Cloud Run when the pipeline is stable.

## Roadmap — research

- **Glass ratio from Street View / satellite imagery ML.** Closes the
  plan §9 R1 gap; multi-month effort with real labeled-data needs.
- **Species-specific weighting.** eBird abundance × species collision
  susceptibility. Requires defensible priority-species selection.
- **Absolute mortality calibration.** Requires a monitoring dataset and
  strong assumption stack; may not be defensible without partner buy-in.

## Limitations (short version)

This tool produces a **relative** risk ranking, not calibrated mortality
counts. It has no glass-area or reflectivity data, and ALAN is 500m
resolution. Overture heights undercount tall antennas/superstructures.
See [docs/methodology.md](docs/methodology.md) for the full list.

## Repo layout

```
config.yaml                  # city bboxes, weights, GEE project, model params
pyproject.toml               # uv/pip project manifest
pipeline/
  buildings.py               # T2 — Overture load, clean, height backfill, geometry
  alan.py                    # T3 — Black Marble composite + batched sampling
  habitat.py                 # v2A — OSM habitat fetch + distance + edge multiplier
  risk.py                    # T4 — v2 index: sigmoid H, class multiplier, edge factor
  scenario.py                # T5 — lighting-reduction scenario engine
  config.py                  # config.yaml loader
scripts/
  fetch_overture_city.py     # 1. fetch
  build_buildings.py         # 2. clean + geometry
  build_alan.py              # 3. sample nighttime lights
  build_risk.py              # 4. score
  export_geojson.py          # 5. kepler-ready export
  run_scenario.py            # T5 demo CLI
  check_gee_auth.py          # setup smoke test
  inspect_local_footprints.py # dev inspection
notebooks/
  validation_cbcm.ipynb      # CBCM ground-truth validation (GitHub-rendered)
  _build_validation_notebook.py  # regenerator — run to rebuild the .ipynb
tests/
  test_risk.py               # 22 tests covering v1 + v2 risk math
  test_scenario.py           # 7 tests covering scenario engine
docs/
  methodology.md             # model spec + citations
  phase2_birdflow.md         # BirdFlow design notes stub
  README.md                  # kepler HTML update workflow
data/
  cities/                    # raw Overture GeoJSON per city
  processed/                 # cleaned + scored intermediates
```
