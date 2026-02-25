#!/usr/bin/env python3
"""Check sensitivity stability versus output NSIDE for preprocessing.

This utility re-runs preprocessing for one map multiple times while varying only
``nside_out`` in the ``smooth_maps`` operation, then reports derived
``uK-arcmin`` sensitivities from the requested variance field.

Typical use:
    python src/mde_pipeline/qc/check_preprocess_sensitivity_stability.py \
        --map-id planck353_pr3 --field QQ --nside-out 16 --nside-out 32 --nside-out 64

Interpretation:
- If ``target_fwhm_arcmin`` is held fixed, derived ``uK-arcmin`` should be fairly
  stable versus ``nside_out`` (small pixelization differences are expected).
- Large changes versus NSIDE suggest an issue in variance propagation/down/up-grading.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import healpy as hp
import numpy as np
import yaml

from mde_pipeline.io import raw_maps_readers
from mde_pipeline.preprocessing.beams import smooth_maps
from mde_pipeline.preprocessing.units import dec_mask, fix_pol_convention, subtract_cmb, unit_convert


OPS = {
    "fix_pol_convention": fix_pol_convention,
    "unit_convert": unit_convert,
    "dec_mask": dec_mask,
}

OP_FACTORIES = {
    "subtract_cmb": subtract_cmb,
}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def valid_pixels(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr) & (arr > -1e20)


def unit_to_uK_factor(unit: str) -> float:
    u = (unit or "").strip().lower()
    if u.startswith("uk"):
        return 1.0
    if u.startswith("mk"):
        return 1e3
    if u.startswith("k"):
        return 1e6
    raise ValueError(f"Unsupported map unit for sensitivity conversion: {unit!r}")


def derive_sensitivity_uK_arcmin(var_map: np.ndarray, unit: str) -> Tuple[float, int, int]:
    good = valid_pixels(var_map)
    if not np.any(good):
        raise ValueError("No valid pixels in variance map")

    nside = hp.npix2nside(var_map.size)
    median_var_native = float(np.median(var_map[good]))
    to_uK = unit_to_uK_factor(unit)
    median_var_uK2 = median_var_native * (to_uK ** 2)
    omega_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 60.0 * 60.0
    sensitivity = float(np.sqrt(median_var_uK2 * omega_arcmin2))
    return sensitivity, nside, int(np.count_nonzero(good))


def _build_operations() -> Dict[str, Any]:
    ops = dict(OPS)
    for op_name, factory in OP_FACTORIES.items():
        candidate = factory()
        if not callable(candidate):
            raise TypeError(f"Factory for op '{op_name}' did not return callable")
        ops[op_name] = candidate
    return ops


def _load_raw_map(map_id: str, instruments: Dict[str, Any]):
    meta = instruments[map_id]
    read_class = meta.get("read_class")
    if not read_class:
        raise ValueError(f"[{map_id}] missing read_class")
    if not hasattr(raw_maps_readers, read_class):
        raise ValueError(f"[{map_id}] reader class not found: {read_class}")

    reader_cls = getattr(raw_maps_readers, read_class)
    reader = reader_cls(map_id, meta)
    return reader


def _prepare_steps(preprocess_cfg: Dict[str, Any], map_id: str) -> List[dict]:
    return list(preprocess_cfg.get(map_id, preprocess_cfg.get("_default", [])))


def _extract_base_smoothing(step_list: Iterable[dict]) -> Tuple[float | None, int | None]:
    target_fwhm = None
    target_nside = None
    for step in step_list:
        if step.get("op") == "smooth_maps":
            target_fwhm = step.get("target_fwhm_arcmin")
            target_nside = step.get("nside_out")
            break
    return target_fwhm, target_nside


def _apply_non_smoothing_steps(reader, steps: List[dict]) -> None:
    ops = _build_operations()
    for step in steps:
        op_name = step.get("op")
        if op_name == "smooth_maps":
            continue
        if op_name not in ops:
            raise KeyError(f"Unknown preprocess op: {op_name}")
        kwargs = {k: v for k, v in step.items() if k != "op"}
        ops[op_name](reader.map, reader.beam_info, **kwargs)


def _field_array(m, field: str) -> np.ndarray:
    arr = getattr(m, field, None)
    if not isinstance(arr, np.ndarray) or arr.size == 0:
        raise ValueError(f"Field {field} not present")
    return arr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map-id", required=True, help="Map ID in instruments.yaml, e.g. planck353_pr3")
    p.add_argument("--instruments", type=Path, default=Path("configs/preprocessing/instruments.yaml"))
    p.add_argument("--preprocess", type=Path, default=Path("configs/preprocessing/preprocessing.yaml"))
    p.add_argument("--field", choices=["II", "QQ", "UU"], default="QQ")
    p.add_argument(
        "--nside-out",
        type=int,
        action="append",
        default=[],
        help="Target nside_out for smooth_maps; can be repeated (defaults to [16, 32, 64, 128]).",
    )
    p.add_argument(
        "--target-fwhm-arcmin",
        type=float,
        default=None,
        help="Override target_fwhm_arcmin for smooth_maps (defaults to preprocessing.yaml value).",
    )
    args = p.parse_args()

    instruments = load_yaml(args.instruments)
    preprocess_cfg = load_yaml(args.preprocess)

    if args.map_id not in instruments:
        raise KeyError(f"map-id {args.map_id!r} not found in {args.instruments}")

    steps = _prepare_steps(preprocess_cfg, args.map_id)
    base_fwhm, base_nside = _extract_base_smoothing(steps)

    if args.target_fwhm_arcmin is not None:
        target_fwhm = float(args.target_fwhm_arcmin)
    elif base_fwhm is not None:
        target_fwhm = float(base_fwhm)
    else:
        raise ValueError("No smooth_maps step found and --target-fwhm-arcmin not provided.")

    nside_values = list(args.nside_out) if args.nside_out else [16, 32, 64, 128]

    # Raw baseline
    raw_reader = _load_raw_map(args.map_id, instruments)
    raw_arr = _field_array(raw_reader.map, args.field)
    raw_sens, raw_nside, raw_ngood = derive_sensitivity_uK_arcmin(raw_arr, raw_reader.map.unit)

    print(f"map_id={args.map_id} field={args.field} unit={raw_reader.map.unit}")
    print(f"raw_nside={raw_nside} raw_good_pixels={raw_ngood} raw_uKarcmin={raw_sens:.6f}")
    print(f"target_fwhm_arcmin={target_fwhm} (config nside_out={base_nside})")
    print()

    header = (
        f"{'nside_out':>9s} {'processed_uKarcmin':>18s} {'ratio_to_raw':>13s} "
        f"{'ratio_to_first':>14s} {'good_pixels':>12s}"
    )
    print(header)
    print("-" * len(header))

    first = None
    for nside_out in nside_values:
        reader = _load_raw_map(args.map_id, instruments)
        _apply_non_smoothing_steps(reader, steps)

        smooth_maps(
            reader.map,
            reader.beam_info,
            target_fwhm_arcmin=target_fwhm,
            nside_out=int(nside_out),
        )

        arr = _field_array(reader.map, args.field)
        proc_sens, _, ngood = derive_sensitivity_uK_arcmin(arr, reader.map.unit)

        if first is None:
            first = proc_sens
        ratio_raw = proc_sens / raw_sens if raw_sens != 0 else np.nan
        ratio_first = proc_sens / first if first != 0 else np.nan

        print(f"{nside_out:9d} {proc_sens:18.6f} {ratio_raw:13.6f} {ratio_first:14.6f} {ngood:12d}")


if __name__ == "__main__":
    main()
