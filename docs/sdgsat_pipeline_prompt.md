# Claude Code handoff — SDGSAT-1 high-res ALAN experiment

Copy everything below the line into Claude Code from the repo root. It is
scoped to the **MVP experiment** in [docs/sdgsat_alan.md](sdgsat_alan.md)
(parallel sampling + ranking diff), NOT full VIIRS×SDGSAT fusion — that comes
only if the experiment shows the leaderboard reorders.

---

## Context

This repo scores relative bird–window collision risk per building for a city.
The lighting term `L` currently comes from 500 m VIIRS Black Marble
(`pipeline/alan.py` → `scripts/build_alan.py`, output column `alan_radiance`).
At 500 m every building in a pixel shares one radiance value, and because `L`
enters the risk index multiplicatively (`pipeline/risk.py`:
`Risk_raw = structure_score × norm(alan_radiance) × class_multiplier × edge_factor`),
an isolated bright tower in a dark pixel gets `norm(L)≈0` and its structural
hazard is wiped out. Read `docs/methodology.md` and `docs/sdgsat_alan.md`
first — the second is the design note this task implements.

We want to test whether **SDGSAT-1 Glimmer** (40 m RGB incl. a blue band,
10 m panchromatic) changes which buildings rank highest. Do NOT rip out
VIIRS — add SDGSAT as a parallel lighting source and compare.

## Guardrails (match existing conventions)

- Python 3.11, `uv`. Mirror the style of `pipeline/alan.py` and
  `pipeline/habitat.py`: small pure functions, type hints, docstrings citing
  the "why," null-preserving, no network calls inside `pipeline/` core funcs
  where avoidable (fetch in `scripts/`).
- Config-driven: new params go in `config.yaml`, loaded via `pipeline/config.py`.
  Do not hardcode paths or scene IDs in code.
- Every model-math function gets a `pytest` unit test on a tiny synthetic
  frame (known inputs → known outputs), like `tests/test_risk.py`.
- Keep the existing pipeline working unchanged. SDGSAT is additive.
- Keep the risk model's public interface intact — the experiment must be
  runnable by swapping which `L` column `pipeline/risk.py` reads, not by
  forking the scoring logic.

## Tasks (do in order; each ends with a runnable artifact + a note on assumptions)

**S1 — Config + data access.**
Add an `assets.alan_hires` block to `config.yaml`: SDGSAT-1 product/band
names, per-city scene list (leave Chicago scene IDs as a `TODO` placeholder
with a comment on where to get them), a `bands` map (pan, blue, green, red
with their nm ranges), and a `blend` stub for later. SDGSAT-1 has **no Earth
Engine ImageCollection** — data is scene files from the CBAS Open Science
Program (registration required). Write `scripts/fetch_sdgsat.py` that, given
a local directory of downloaded Glimmer scene GeoTIFFs (I will download them
manually — do NOT attempt to script the CBAS portal login), validates them,
reports band/CRS/extent/coverage over the active city's `dev_bbox`, and
mosaics the best cloud-free scene(s) for the migration window into one
raster. If no scenes are present, exit with a clear "download scenes to
`data/sdgsat/<city>/` first" message. Print a coverage summary.

**S2 — `pipeline/alan_hires.py` (sampling).**
New module. Given a buildings GeoDataFrame and a SDGSAT raster, sample
radiance per building **centroid** for panchromatic and for the blue band,
in the city's projected CRS. Return columns `alan_sdgsat` (pan) and
`alan_sdgsat_blue`. Preserve nulls where a building falls outside scene
coverage (do NOT silently fill — these buildings should fall back to VIIRS
downstream). Mirror the null-handling philosophy in `pipeline/risk.py`.
Radiometric calibration: apply the scene's documented scale/offset to get
physical radiance; document the exact conversion in the docstring.

**S3 — `scripts/build_alan_hires.py`.**
Reads `data/processed/<city>_buildings_dev_alan.gpkg` (VIIRS output) + the
S1 mosaic, calls `pipeline/alan_hires.py`, writes
`data/processed/<city>_buildings_dev_alan_hires.gpkg` with both the VIIRS
`alan_radiance` and the new `alan_sdgsat*` columns. Log: % buildings covered
by SDGSAT, correlation between VIIRS and SDGSAT radiance, and how many fell
back to VIIRS.

**S4 — Risk model: selectable L source (small, surgical).**
In `pipeline/risk.py` / `config.yaml`, add a config key `lighting_source`
∈ {`viirs`, `sdgsat_pan`, `sdgsat_blue`}. `compute()` reads the chosen column
into the existing `norm(L)` slot — everything else in the index is unchanged.
Where the chosen SDGSAT column is null for a building, fall back to
`alan_radiance` (VIIRS) so no building drops out. Add unit tests covering the
fallback and the three source options.

**S5 — Comparison harness (the actual experiment).**
`scripts/compare_lighting.py`: run the full risk model three times
(`viirs`, `sdgsat_pan`, `sdgsat_blue`) on the same buildings, and emit:
- top-25 leaderboard for each, side by side;
- rank-correlation (Spearman) between VIIRS and each SDGSAT variant;
- the buildings that move most (biggest rank jumps), with their
  height/footprint and both radiance values;
- a one-paragraph auto-summary: did the leaderboard reorder materially?
Write results to `data/processed/lighting_comparison.csv` and print the
summary. This is the deliverable that decides whether fusion is worth it.

**S6 — Docs + verification (do not skip).**
Update `docs/sdgsat_alan.md` "Open decisions" with what you found. In
`docs/methodology.md`, note the experiment and its result. Verification
checklist: (a) VIIRS pipeline still passes existing tests unchanged; (b)
SDGSAT sampling values are physically plausible (spot-check 3 buildings
against the raster in QGIS or via a printed pixel lookup); (c) fallback
logic leaves zero buildings unscored; (d) reducing to a single-scene mosaic
didn't silently drop the Loop. Report all four.

## Definition of done

- `uv run python scripts/build_alan_hires.py` and
  `uv run python scripts/compare_lighting.py` run end-to-end on the Chicago
  dev bbox once scenes are downloaded.
- New unit tests pass; existing 29 tests still pass.
- `lighting_comparison.csv` + printed summary answer: does 40 m + blue
  reorder the top-N vs 500 m VIIRS?
- No change to default behavior when `lighting_source: viirs` (the default).

## Explicitly out of scope for this task

- Formal VIIRS×SDGSAT spatiotemporal fusion (only if S5 justifies it).
- Scripting the CBAS download/login — scenes are fetched manually.
- Jilin-1 / ISS imagery (validation-only, separate effort).
- Any frontend/kepler changes.
