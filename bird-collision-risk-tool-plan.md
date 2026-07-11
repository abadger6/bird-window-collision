# Bird–Window Collision Risk Tool — Build Plan & Handoff Spec

**Version:** 1.0
**Prepared for:** Claude Code (implementation handoff)
**Author context:** Product/strategy owner (Alex)
**Status:** Ready to start Phase 1

---

## 0. TL;DR for the implementer

Build a geospatial tool that produces a **relative bird–collision risk score for every building in a city**, driven in Phase 1 by two data layers: **building geometry** (footprint, height) and **artificial light at night / ALAN** (VIIRS nighttime radiance). Render it as an interactive map where advocates and policymakers can (a) see a risk heatmap, (b) inspect and rank individual buildings, and (c) run a **"lights-out" mitigation scenario** to see how much risk drops if lighting is reduced in a chosen area.

Phase 2 (designed for now, built later) adds a **BirdFlow-derived migration layer** so risk is weighted by *when and where birds are actually moving*, not just where buildings are bright.

**Pilot city:** Chicago (configurable). **Frontend:** Google Earth Engine App for v1 (see §3 for the important caveat on decoupling the compute from the UI).

---

## 1. Goal & users

**Goal.** Help advocates and policymakers identify the **highest-impact protective actions** to reduce bird–building collisions, by mapping relative collision risk and letting them test interventions.

**Primary users.** Conservation advocates, city sustainability offices, Lights Out program organizers, building-policy staff. Assume GIS-literate but not developers. They need a shareable web link, not a codebase.

**Core questions the tool must answer:**
1. Where are the highest-risk buildings/blocks in this city?
2. If we got building X (or district Y) to reduce lighting by Z%, how much collision risk would that remove?
3. Which interventions give the most risk reduction per building treated? (prioritization)

**Non-goals (v1):** real-time nightly forecasting, per-species targeting, individual-bird tracking, guaranteed-absolute mortality counts. This is a **relative risk / prioritization** tool, not a calibrated mortality predictor.

---

## 2. Scope & phasing

### Phase 1 — Hazard map (ALAN + buildings) — THIS BUILD
- Ingest building footprints + heights for the pilot city.
- Ingest VIIRS ALAN and sample radiance per building.
- Compute a **relative structural-hazard × lighting-environment risk index** per building.
- Interactive map + ranked table + CSV/GeoJSON export.
- Lighting-reduction scenario tool.

### Phase 1.5 — Habitat-edge modifier (cheap, optional add-on)
- Add proximity-to-greenspace/water as a risk multiplier (birds concentrate near habitat edges). Uses land-cover raster + OSM parks/water. Low effort, well-grounded in literature.

### Phase 2 — Ecological weighting (BirdFlow) — DESIGNED NOW, BUILT LATER
- Generate weekly species/aggregate **migration-traffic surfaces** from eBird Status & Trends via the BirdFlow framework (offline, Python).
- Ingest as GEE raster assets and fold into the risk index as a temporal/spatial weight.
- Add a season/week selector to the UI so risk becomes time-aware.

### Explicitly out of scope (all phases, for now)
- Façade glass-area / reflectivity attributes (not available in open data — see §9 Risk R1).
- Real-time radar (BirdCast) operational alerts.
- Native mobile app.

---

## 3. Architecture decision (and the GEE pushback you asked for)

**Recommendation: build v1 as a GEE App, but decouple the model from the UI from day one.**

Your instinct to host on GEE is reasonable for Phase 1 — the computation (rasterize ALAN, sample radiance per building, weighted index, zonal stats) sits squarely in GEE's sweet spot, and a GEE App is the fastest path to a shareable public link with zero server ops. Go for it.

**The caveat / honest pushback.** GEE Apps have real limits you will hit as this grows:
- **Large interactive vector layers are painful.** A full city is 10⁵–10⁶ building polygons. GEE Apps can render and click-query these, but interactivity gets sluggish and you can bump into per-tile memory limits and app quota (429) errors under public load.
- **UI is basic.** The `ui.*` widget set is limited; no custom components, limited styling, clunky for polished scenario sliders and rich per-building popups.
- **Not a modeling engine.** Fine for a weighted index; a poor home for anything statistical/probabilistic. Phase 2's BirdFlow modeling **must** happen offline regardless.

**Therefore, structure the system so the compute is portable:**

```
[ Offline pipeline (Python, GEE Python API) ]
   -> produces a scored building FeatureCollection as a GEE asset
        |
        +--> Option A (v1): thin GEE App reads the asset, renders map + widgets   <-- START HERE
        |
        +--> Option B (later): export vector tiles / GeoJSON, serve in a
             MapLibre/deck.gl web map for better UX + scenario interactivity
```

Because the **scored-building asset** is the interface, you can swap the frontend from a GEE App to a custom web map later **without touching the model**. Build Option A now; keep Option B cheap to reach. Do **not** put the risk computation inline in the App's client code — put it in the offline pipeline and have the App consume precomputed results.

---

## 4. Data sources

> Verify exact GEE asset IDs at build time from the [Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets) and the [awesome-gee-community-catalog](https://gee-community-catalog.org/). IDs below are current-best; confirm before hardcoding.

### 4.1 Buildings (footprints + heights)
- **Primary (US): Overture Maps buildings** — includes LiDAR-derived heights for the USA and is available as a community dataset in GEE. Confirm the current asset path via the awesome-gee-community-catalog "Overture buildings" project page.
- **Fallback / supplement: Microsoft US Building Footprints** — ~125M US polygons (GeoJSON on GitHub); not natively in the GEE catalog, so ingest the pilot-city subset as a GEE asset if Overture coverage is thin.
- **OSM buildings** — useful for `building:levels`, `building:use`, and to backfill height (levels × ~3 m) where Overture height is null.
- **Height handling:** prefer Overture height → else OSM `building:levels`×3 m → else flag `height = null` and exclude from height-weighted term (document this).

### 4.2 Lighting (ALAN)
- **VIIRS DNB monthly composites:** `NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG` (stray-light corrected) or `.../VCMCFG`. ~500 m (15 arc-sec) resolution.
- **VIIRS DNB annual composites:** `NOAA/VIIRS/DNB/ANNUAL_V22` for a stable multi-year mean.
- **NASA Black Marble daily (optional, finer temporal):** `NASA/VIIRS/002/VNP46A2` (gap-filled, lunar/atmosphere-corrected), ~500 m.
- **Resolution reality check:** ALAN is ~500 m/pixel — a *district-scale* brightness proxy, NOT per-building lighting. Every building within a 500 m pixel inherits the same radiance. Treat ALAN as the "lighting environment" term, and be explicit about this in the UI/docs. Use a **multi-month or annual mean** to reduce noise; consider seasonal (spring/fall migration months) composites.

### 4.3 Habitat edge (Phase 1.5)
- **Land cover:** `ESA/WorldCover/v200` (10 m global) or US `USGS/NLCD_RELEASES/...` (NLCD).
- **Parks/water:** OSM `leisure=park`, `natural=water`, or land-cover-derived. Compute distance-to-greenspace/water raster.

### 4.4 Birds (Phase 2)
- **eBird Status & Trends** weekly relative abundance (GeoTIFF, ~3 km) via the `ebirdst` R package (requires free data-access request).
- **BirdFlow** (`BirdFlowR` / `BirdFlowPy`) to convert abundance into weekly movement/migration-traffic surfaces. Runs offline (GPU helpful). Output: weekly migration-intensity rasters ingested as GEE assets.

---

## 5. Risk model (Phase 1)

A transparent, **relative** index — not a black box — so advocates can trust and explain it. All terms normalized within the city (percentile or min–max) to 0–1.

### 5.1 Per-building terms
| Term | Definition | Rationale (literature) |
|---|---|---|
| **Façade exposure** `F` | `perimeter × height` (proxy for glass surface area) | Collision risk scales with façade/glass area; taller, larger buildings kill more. |
| **Footprint** `A` | building footprint area | Larger buildings = larger target. |
| **Height** `H` | building height (m) | Height is an independent structural predictor. |
| **Lighting env** `L` | mean VIIRS radiance sampled at building centroid/footprint (500 m) | Lighted area is often a better predictor than glass area; ALAN correlates with collisions. |
| **Habitat edge** `E` *(Phase 1.5)* | inverse distance to greenspace/water | Collisions concentrate near habitat edges/green space. |

### 5.2 Index formula (default; weights configurable)
```
Structure_i = w_F·norm(F_i) + w_A·norm(A_i) + w_H·norm(H_i)      # structural hazard
Risk_raw_i  = Structure_i × norm(L_i)                             # × lighting environment
Risk_i      = 100 × percentile_rank(Risk_raw_i)                   # 0–100 relative score
# Phase 1.5: multiply Risk_raw_i by (1 + w_E·norm(E_i))
```
**Default weights:** `w_F=0.5, w_A=0.2, w_H=0.3` (structure), `w_E=0.5` (edge). Expose all as config. The multiplicative `× L` encodes "a big glassy building in a dark area and a bright area are not equally dangerous."

### 5.3 Scenario engine (the policymaker feature)
Let the user select buildings or draw an area and set a **lighting reduction %** `r`. Recompute `L' = L × (1 − r)` for affected buildings, recompute `Risk`, and report:
- Total city risk removed (Σ Risk before − after).
- Risk removed per building treated (efficiency ranking → "highest-impact actions").
- Before/after map + summary stats.

This directly answers "where does mitigation have the highest impact." Keep the computation lightweight enough to run interactively (precompute base terms; scenario only rescales `L`).

**Important — do scenario comparisons on `Risk_raw`, not `Risk`.** The percentile-ranked `Risk` (§5.2) is a *display* quantity: rescaling `L` for a subset of buildings shifts everyone else's rank purely by relative motion, which makes "risk removed" non-additive and can produce misleading before/after deltas for untreated buildings. Compute totals, per-building deltas, and efficiency rankings on `Risk_raw`; re-percentile only for the visualization.

### 5.4 Outputs
- Choropleth/graduated building layer colored by `Risk`.
- Ranked table of top-N risk buildings (with lat/lon, height, radiance, score).
- Export: GeoJSON + CSV of scored buildings.
- Scenario report (JSON/CSV).

---

## 6. Implementation plan (Phase 1 tasks for Claude Code)

Work in this order. Each task should end with a runnable artifact and a short note on assumptions/limitations.

**T1 — Repo + environment setup**
- Python 3.11+, `earthengine-api`, `geemap`, `geopandas`, `osmnx`, `pandas`. Authenticate GEE (service account or user). Create the repo structure in §7. Add a `config.yaml` (pilot city bbox/name, weights, ALAN product, date range).

**T2 — Building data pipeline**
- Load Overture buildings for the city bbox; attach heights. Backfill missing heights from OSM `building:levels`. Clean geometries, compute `footprint_area`, `perimeter`, `facade_area = perimeter × height`. Output a clean building FeatureCollection (GEE asset) + local GeoDataFrame. Log coverage/height-null stats.

**T3 — ALAN pipeline**
- Load VIIRS product, build a spring+fall migration-months composite (mean radiance) for the city. Sample radiance per building (`reduceRegions`, mean). Handle the 500 m resolution honestly (nearest-pixel is fine). Attach `alan_radiance` to buildings.

**T4 — Risk index**
- Implement §5 normalization + weighting + percentile rank. Parameterize weights from `config.yaml`. Attach `structure_score`, `risk_raw`, `risk_score` to buildings. Unit-test the math on a tiny synthetic set (known inputs → known outputs).

**T5 — Scenario engine**
- Function: given building IDs/area + reduction %, recompute risk and return before/after totals and per-building efficiency ranking. Pure function over the precomputed base terms so it runs fast.

**T6 — GEE App frontend (Option A)**
- Publish the scored building asset. Build a GEE App: map with graduated risk styling, legend, click-to-inspect popup (height, radiance, score), top-N table panel, a draw-area + reduction-slider scenario control, and a "download results" affordance. Enable caching. Keep all heavy compute out of the client — App only reads the precomputed asset + does the lightweight `L`-rescale for scenarios.

**T7 — Exports & docs**
- GeoJSON/CSV export of scored buildings + scenario results. Write a README: data sources, model formula, weight rationale, **explicit limitations** (relative not absolute; ALAN 500 m; no glass data), and how to re-run for another city.

**T8 — Verification (do not skip)**
- Sanity checks: do known-dangerous Chicago buildings (e.g., McCormick Place / large lakefront glass towers) rank high? Spot-check radiance sampling against a VIIRS basemap. Confirm scenario math (reducing lighting can only lower or hold risk). Cross-check building count vs. an independent source. Document results.

---

## 7. Repo structure

```
bird-collision-risk/
  config.yaml                 # city, bbox, weights, product IDs, date ranges
  requirements.txt
  README.md
  pipeline/
    __init__.py
    buildings.py              # T2: fetch/clean footprints + heights
    alan.py                   # T3: VIIRS composite + per-building sampling
    risk.py                   # T4: normalization, weighting, scoring
    scenario.py               # T5: lighting-reduction recompute
    export.py                 # T7: GeoJSON/CSV
    ee_assets.py              # asset upload/publish helpers
  app/
    gee_app.js                # T6: Earth Engine App UI (JS)  <-- Option A
    README_app.md
  tests/
    test_risk.py              # T4 unit tests
    test_scenario.py          # T5 unit tests
  notebooks/
    01_explore_buildings.ipynb
    02_explore_alan.ipynb
  docs/
    methodology.md            # model + limitations
    phase2_birdflow.md        # §8 design
```

---

## 8. Phase 2 design (BirdFlow) — build later, keep the seams

- **Offline:** request eBird S&T access; run BirdFlow (`BirdFlowPy`) to produce **weekly migration-traffic rasters** for the region. Reproject/resample to a working grid; ingest as a GEE ImageCollection (one image per week).
- **Model change:** add a temporal weight `M_{i,week}` = normalized migration traffic at building `i` for that week. New index:
  `Risk_{i,week} = Structure_i × norm(L_i) × (α + β·norm(M_{i,week}))`.
  Defaults e.g. `α=0.25, β=1.0` so risk never zeroes out off-peak but scales strongly with migration.
- **UI change:** add a **week/season selector**; risk map animates across the migration calendar. Scenario tool can now target *peak-migration weeks* → far more compelling "highest-impact timing" story for Lights Out.
- **Resolution note to carry forward:** BirdFlow inherits eBird's ~3 km grid, so the ecological weight is coarse while building geometry stays sharp. Design the legend/UX to communicate "sharp hazard geometry, coarse ecological weighting."
- **Why this ordering is low-risk:** BirdFlow is generated from eBird data you control, so it avoids the BirdCast bulk-data dependency; and because the model already multiplies independent normalized terms, adding `M` is a clean extension, not a rewrite.

---

## 9. Risks & assumptions

- **R1 — No façade/glass data (structural).** Open building data lacks glass area/reflectivity, a top collision driver. Mitigation: `facade_area = perimeter × height` proxy; document that risk is structural-geometry-based. Future: street-imagery ML to estimate glazing.
- **R2 — ALAN is district-scale (500 m).** Cannot distinguish lit vs. dark neighboring buildings. Mitigation: treat as "lighting environment," not building lighting; communicate clearly.
- **R3 — Relative, not absolute.** Scores rank buildings; they are not calibrated mortality counts. No open per-building collision-count data to calibrate against (Chicago monitoring data is the best future calibration source). Frame the tool accordingly everywhere.
- **R4 — GEE App scaling/quota.** Public load can trigger 429s; large vector layers can lag. Mitigation: caching, precomputed asset, plan Option B (web map) as the escape hatch.
- **R5 — Building data completeness/height nulls.** Coverage varies. Mitigation: multi-source backfill + logged coverage stats; exclude nulls transparently.
- **Assumptions:** pilot = Chicago; US-focused datasets; GEE account with asset-publishing rights available; users want a shareable web link, not a repo.

---

## 10. Tech stack summary

- **Compute/model:** Python + Earth Engine Python API (`earthengine-api`, `geemap`), `geopandas`, `osmnx`.
- **Frontend v1:** Earth Engine App (JavaScript, `ui.*`).
- **Frontend v2 (optional later):** MapLibre GL or deck.gl over exported vector tiles/GeoJSON.
- **Phase 2 modeling:** `ebirdst` (R) + `BirdFlowPy` (Python), offline, GPU optional.
- **Testing:** `pytest` for model + scenario math.

---

## 11. Definition of done (Phase 1)

1. Reproducible pipeline that ingests Chicago buildings + ALAN and outputs a scored building asset.
2. Published GEE App: risk heatmap, per-building inspect, top-N ranking, lighting-reduction scenario with before/after impact stats.
3. GeoJSON + CSV exports.
4. README/methodology docs with explicit limitations.
5. Passing unit tests + completed §6 T8 verification checklist.
6. `config.yaml` lets the whole thing re-run for a different city by changing bbox/name.

