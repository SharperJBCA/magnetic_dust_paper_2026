#!/usr/bin/env python3
"""Compare map sensitivities inferred from variance maps against reference μK·arcmin values.

Formula used per requested field (default QQ):
    sensitivity = sqrt(median(variance_field_valid) * Omega_pixel)
where Omega_pixel is in arcmin^2 and variance is converted to μK^2.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import healpy as hp
import numpy as np
import yaml

PR3_FIELD_INDEX = {"II": 0, "QQ": 3, "UU": 5}
COSMO_FIELD_INDEX = {"II": 0, "QQ": 1, "UU": 2}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def unit_to_uK_factor(unit: str) -> float:
    u = (unit or "").strip().lower()
    if u.startswith("uk"):
        return 1.0
    if u.startswith("mk"):
        return 1e3
    if u.startswith("k"):
        return 1e6
    raise ValueError(f"Unsupported map unit for sensitivity conversion: {unit!r}")


def load_references(csv_path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["map_id"]] = row
    return out


def select_map_ids(instruments: dict, family: str, explicit_map_ids: Iterable[str]) -> List[str]:
    if explicit_map_ids:
        return list(explicit_map_ids)

    keys = list(instruments.keys())
    if family == "all":
        return [k for k in keys if k.startswith(("litebird_", "planck", "wmap"))]
    if family == "litebird":
        return [k for k in keys if k.startswith("litebird_")]
    if family == "planck":
        return [k for k in keys if k.startswith("planck")]
    if family == "wmap":
        return [k for k in keys if k.startswith("wmap")]
    raise ValueError(f"Unsupported family: {family}")


def valid_pixels(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr) & (arr > -1e20)


def derive_sensitivity_uK_arcmin(filename: str, field: str, unit: str, map_id: str) -> Tuple[float, int, int]:
    print(filename)
    if 'pr3' in map_id:
        arr = hp.read_map(filename, field=4+PR3_FIELD_INDEX[field], verbose=False)
    elif 'cosmo' in map_id:
        arr = hp.read_map(filename, field=3+COSMO_FIELD_INDEX[field], verbose=False)


    nside = hp.npix2nside(arr.size)
    good = valid_pixels(arr)
    if not np.any(good):
        raise ValueError("No valid pixels in variance map field")

    median_var_native = float(np.median(arr[good]))
    to_uK = unit_to_uK_factor(unit)
    median_var_uK2 = median_var_native * (to_uK ** 2)
    omega_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 60.0 * 60.0
    sensitivity = float(np.sqrt(median_var_uK2 * omega_arcmin2))
    return sensitivity, nside, int(np.count_nonzero(good))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instruments", type=Path, default=Path("configs/preprocessing/instruments.yaml"))
    p.add_argument("--reference", type=Path, default=Path("configs/preprocessing/reference_sensitivities_uK_arcmin.csv"))
    p.add_argument("--field", choices=["II", "QQ", "UU"], default="QQ")
    p.add_argument("--family", choices=["all", "litebird", "planck", "wmap"], default="all")
    p.add_argument("--map-id", action="append", default=[], help="Specific map_id(s) to evaluate.")
    p.add_argument("--rtol", type=float, default=0.20, help="Relative tolerance for PASS/FAIL flag.")
    args = p.parse_args()

    instruments = load_yaml(args.instruments)
    refs = load_references(args.reference)
    map_ids = select_map_ids(instruments, args.family, args.map_id)

    header = (
        f"{'map_id':30s} {'field':4s} {'derived_uKarcmin':>16s} "
        f"{'ref_uKarcmin':>12s} {'frac_diff':>10s} {'status':>8s}"
    )
    print(header)
    print("-" * len(header))

    for map_id in map_ids:
        meta = instruments.get(map_id)
        if meta is None:
            print(f"{map_id:30s} {'-':4s} {'-':>16s} {'-':>12s} {'-':>10s} {'MISSING':>8s}")
            continue

        filename = meta.get("filename")
        if not isinstance(filename, str):
            print(f"{map_id:30s} {args.field:4s} {'-':>16s} {'-':>12s} {'-':>10s} {'SKIP':>8s}")
            continue

        try:
            derived, _, _ = derive_sensitivity_uK_arcmin(filename, args.field, meta.get("unit", ""), map_id)
        except Exception:
            print(f"{map_id:30s} {args.field:4s} {'-':>16s} {'-':>12s} {'-':>10s} {'ERROR':>8s}")
            continue

        ref_row = refs.get(map_id)
        if ref_row is None:
            print(f"{map_id:30s} {args.field:4s} {derived:16.3f} {'-':>12s} {'-':>10s} {'NOREF':>8s}")
            continue

        ref = float(ref_row["reference_uK_arcmin"])
        frac = (derived - ref) / ref if ref != 0 else np.nan
        status = "PASS" if np.isfinite(frac) and abs(frac) <= args.rtol else "WARN"
        print(f"{map_id:30s} {args.field:4s} {derived:16.3f} {ref:12.3f} {frac:10.3%} {status:>8s}")


if __name__ == "__main__":
    main()
