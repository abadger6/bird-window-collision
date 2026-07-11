"""Verify Earth Engine authentication and that key assets resolve.

Run after ``earthengine authenticate``. Exits non-zero if anything fails so
CI / a fresh clone can gate on it. Buildings live in a local GeoJSON now
(fetched by scripts/fetch_overture_city.py) — this script only checks GEE
side.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee  # type: ignore[import-untyped]

from pipeline import config


def main() -> int:
    cfg = config.load()
    project = config.require_gee_project(cfg)

    try:
        ee.Initialize(project=project)
    except Exception as exc:
        print(f"[FAIL] ee.Initialize(project={project!r}): {exc}")
        print("       Run: earthengine authenticate")
        return 1
    print(f"[ ok ] ee.Initialize(project={project!r})")

    alan_id = cfg["assets"]["alan"]
    try:
        ic = ee.ImageCollection(alan_id)
        size = ic.limit(1).size().getInfo()
        print(f"[ ok ] alan: {alan_id}  (probe size={size})")
    except Exception as exc:
        print(f"[FAIL] alan: {alan_id} — {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
