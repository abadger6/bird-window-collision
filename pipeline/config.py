"""Load config.yaml and expose small helpers used across pipeline modules."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config.yaml"


def load(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def require_gee_project(cfg: dict[str, Any]) -> str:
    project = (cfg.get("gee") or {}).get("project")
    if not project:
        raise SystemExit(
            "config.yaml → gee.project is not set. Register a Cloud project for "
            "Earth Engine (https://console.cloud.google.com/earth-engine) and set "
            'gee.project: "ee-yourname" (or the project ID you chose).'
        )
    return project


def active_city(cfg: dict[str, Any]) -> dict[str, Any]:
    name = cfg["active_city"]
    cities = cfg.get("cities") or {}
    if name not in cities:
        raise SystemExit(
            f"config.yaml → active_city={name!r} but no matching entry under `cities:`. "
            f"Available: {sorted(cities.keys())}"
        )
    return {"name": name, **cities[name]}
