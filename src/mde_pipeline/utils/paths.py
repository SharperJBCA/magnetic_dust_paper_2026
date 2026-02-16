from pathlib import Path
from typing import Any, Dict, Optional, List

def _ensure_parent(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

def _resolve_version_dir(base: Path, tag: str) -> Path:
    # base like products/processed_maps, tag like v001
    return base / tag

def _default_processed_h5(tag: str) -> Path:
    return _resolve_version_dir(Path("products/processed_maps"), tag) / "processed_maps.h5"

def _default_regions_h5(tag: str) -> Path:
    return _resolve_version_dir(Path("products/regions"), tag) / "regions.h5"

def _default_sims_h5(tag: str) -> Path:
    return _resolve_version_dir(Path("products/simulations"), tag) / "simulations.h5"

def _default_fits_dir(tag: str) -> Path:
    return _resolve_version_dir(Path("products/fits"), tag)

def _default_combine_fits(tag: str) -> Path:
    return _resolve_version_dir(Path("products/combined"), tag) / "combined_cbass_spass.fits"

def _default_cmb_fits(tag: str, fwhm: float) -> Path:
    return _resolve_version_dir(Path("products/cmb"), tag) / f"smoothed_cmb_map_fwhm{fwhm:2.1f}arcmin.fits"
