"""S1 — Validate + mosaic SDGSAT-1 Glimmer scenes for the active city.

Scans ``data/sdgsat/<city>/`` for locally-downloaded Glimmer GeoTIFFs (PAN
and MSS), validates them, reports band / CRS / extent / coverage over the
city's ``dev_bbox``, and mosaics the pan and mss families into two
migration-window rasters:

    data/sdgsat/<city>/<city>_sdgsat_pan_mosaic.tif
    data/sdgsat/<city>/<city>_sdgsat_mss_mosaic.tif

CBAS SDGSAT-1 has NO Earth Engine ImageCollection — scenes must be
downloaded manually via the CBAS Open Science portal (registration
required). This script does NOT log in to CBAS or fetch anything from the
network; it is a local-file validator + mosaic step.

If no scenes are present the script exits cleanly with instructions.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds
from shapely.geometry import box

from pipeline import config


SUPPORTED_EXTS = {".tif", ".tiff", ".TIF", ".TIFF"}


def find_scenes(scene_dir: Path) -> list[Path]:
    """Return all GeoTIFFs directly under ``scene_dir`` (non-recursive)."""
    if not scene_dir.exists():
        return []
    return sorted(p for p in scene_dir.iterdir() if p.suffix in SUPPORTED_EXTS)


def classify_scene(path: Path, band_cfg: dict) -> str | None:
    """Return 'pan' or 'mss' by matching the filename against configured
    product suffixes (case-insensitive). None if it matches neither."""
    name = path.name.upper()
    pan_suffix = str(band_cfg["pan"]["product_suffix"]).upper()
    blue_suffix = str(band_cfg["blue"]["product_suffix"]).upper()
    if pan_suffix in name:
        return "pan"
    if blue_suffix in name:
        return "mss"
    return None


def _dev_bbox_coverage_pct(
    ds: rasterio.io.DatasetReader,
    dev_bbox_wgs84: tuple[float, float, float, float],
) -> float:
    """% of the dev bbox area (in WGS84 lon-lat degrees, a rough proxy) that
    overlaps the raster footprint. Coarse — we only need "does it cover the
    Loop?" not a rigorous area calc, so lon-lat area is fine."""
    minx, miny, maxx, maxy = dev_bbox_wgs84
    dev_poly = box(minx, miny, maxx, maxy)
    if dev_poly.area == 0:
        return 0.0
    # Raster bounds → WGS84 so we can compare like-for-like.
    r_left, r_bottom, r_right, r_top = transform_bounds(
        ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21
    )
    raster_poly = box(r_left, r_bottom, r_right, r_top)
    inter = raster_poly.intersection(dev_poly).area
    return 100.0 * inter / dev_poly.area


def validate_scene(path: Path, dev_bbox_wgs84: tuple[float, float, float, float]) -> dict:
    """Open a scene and return a small dict of {crs, bands, coverage_pct, …}.
    Never raises — errors are reported as ``{'error': str}`` so the caller can
    keep going and print a per-file table."""
    try:
        with rasterio.open(path) as ds:
            return {
                "path": path,
                "crs": str(ds.crs),
                "bands": ds.count,
                "dtype": ds.dtypes[0] if ds.count else "?",
                "width": ds.width,
                "height": ds.height,
                "res_m": (abs(ds.transform.a), abs(ds.transform.e)),
                "bounds": tuple(ds.bounds),
                "coverage_pct": _dev_bbox_coverage_pct(ds, dev_bbox_wgs84),
                "nodata": ds.nodata,
            }
    except Exception as exc:  # noqa: BLE001 — reporter is deliberately broad
        return {"path": path, "error": str(exc)}


def _print_scene_table(scenes: list[dict]) -> None:
    print(f"{'file':50s}  {'kind':4s}  {'bands':>5s}  {'res(m)':>8s}  "
          f"{'cov%':>6s}  {'crs':<12s}")
    for s in scenes:
        name = s["path"].name
        if "error" in s:
            print(f"  {name[:48]:50s}  ERROR: {s['error']}")
            continue
        rx, ry = s["res_m"]
        res_s = f"{rx:.1f}x{ry:.1f}"
        print(f"  {name[:48]:50s}  {s.get('kind', '?'):4s}  {s['bands']:>5d}  "
              f"{res_s:>8s}  {s['coverage_pct']:>5.1f}%  {s['crs']:<12s}")


def mosaic_scenes(scene_paths: list[Path], out_path: Path) -> tuple[int, int]:
    """Mosaic scenes into ``out_path``. Returns (width, height) of the mosaic.

    Uses rasterio.merge with the default first-hit strategy; for cloud-free
    scenes over the same city this is equivalent to a pick-any. If clouds
    matter you should pre-mask before running this step (out of scope for MVP).
    """
    srcs = [rasterio.open(p) for p in scene_paths]
    try:
        mosaic, transform = merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=mosaic.shape[0],
            compress="deflate",
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mosaic)
        return mosaic.shape[2], mosaic.shape[1]
    finally:
        for s in srcs:
            s.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=None,
        help="Override the default data/sdgsat/<city>/ scene directory",
    )
    args = parser.parse_args()

    cfg = config.load()
    city = config.active_city(cfg)
    band_cfg = cfg["assets"]["alan_hires"]["bands"]
    dev_bbox = tuple(city["dev_bbox"])

    scene_dir = args.scene_dir or (
        Path(cfg["output"]["local_dir"]) / "sdgsat" / city["name"]
    )
    print(f"SDGSAT validator — city={city['name']}")
    print(f"  scene dir = {scene_dir}")
    print(f"  dev bbox  = {dev_bbox}")

    scene_files = find_scenes(scene_dir)
    if not scene_files:
        print("")
        print(f"[skip] No SDGSAT scenes found in {scene_dir}.")
        print("       Download Glimmer scenes manually from the CBAS Open Science")
        print("       portal (https://www.sdgsat.ac.cn) and drop the .tif files here.")
        print("       See config.yaml → cities.<city>.sdgsat_scenes for search hints.")
        return 0

    scenes = [validate_scene(p, dev_bbox) for p in scene_files]
    for s in scenes:
        if "error" not in s:
            s["kind"] = classify_scene(s["path"], band_cfg) or "?"
    _print_scene_table(scenes)

    good = [s for s in scenes if "error" not in s and s.get("coverage_pct", 0) > 0]
    pan = [s["path"] for s in good if s.get("kind") == "pan"]
    mss = [s["path"] for s in good if s.get("kind") == "mss"]

    print("")
    print(f"  {len(pan)} PAN scenes, {len(mss)} MSS scenes usable "
          f"(coverage>0% over dev bbox).")
    if not pan and not mss:
        print("[skip] No scenes cover the dev bbox — nothing to mosaic.")
        return 0

    out_dir = scene_dir
    if pan:
        out_pan = out_dir / f"{city['name']}_sdgsat_pan_mosaic.tif"
        w, h = mosaic_scenes(pan, out_pan)
        print(f"  wrote {out_pan.name}  ({w}x{h}px)")
    if mss:
        out_mss = out_dir / f"{city['name']}_sdgsat_mss_mosaic.tif"
        w, h = mosaic_scenes(mss, out_mss)
        print(f"  wrote {out_mss.name}  ({w}x{h}px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
