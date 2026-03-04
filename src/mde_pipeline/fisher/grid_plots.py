from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..utils.logging import get_logger

log = get_logger(__name__)

_JOB_RE = re.compile(r"^(j\d{4})_(.+)$")
_TOKEN_RE = re.compile(r"([A-Za-z0-9]+)([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)$")


@dataclass
class GridRecord:
    run_dir: Path
    job_id: str
    params: Dict[str, float]


def _parse_params_from_job_dirname(name: str) -> Optional[Tuple[str, Dict[str, float]]]:
    match = _JOB_RE.match(name)
    if not match:
        return None
    job_id = match.group(1)
    suffix = match.group(2)
    params: Dict[str, float] = {}
    for token in suffix.split("_"):
        m = _TOKEN_RE.match(token)
        if not m:
            continue
        key = m.group(1)
        try:
            params[key] = float(m.group(2))
        except ValueError:
            continue
    return job_id, params


def load_grid_manifest(runs_root: Path, manifest_path: Optional[Path] = None) -> List[GridRecord]:
    """
    Manifest format (JSON):
    {
      "jobs": {
        "j0001": {"run_name": "j0001_A0.1_chi01", "params": {"A_md": 0.1, "chi0": 1.0}},
        ...
      }
    }
    """
    if manifest_path is not None:
        raw = json.loads(manifest_path.read_text())
        jobs = raw.get("jobs", {})
        records: List[GridRecord] = []
        for job_id, meta in jobs.items():
            run_name = str(meta.get("run_name", job_id))
            params = {str(k): float(v) for k, v in dict(meta.get("params", {})).items()}
            records.append(GridRecord(run_dir=runs_root / run_name, job_id=str(job_id), params=params))
        return records

    records = []
    for child in sorted(runs_root.iterdir()):
        if not child.is_dir():
            continue
        parsed = _parse_params_from_job_dirname(child.name)
        if parsed is None:
            meta_path = child / "grid_point.json"
            if not meta_path.exists():
                continue
            raw = json.loads(meta_path.read_text())
            job_id = str(raw.get("job_id", child.name))
            params = {str(k): float(v) for k, v in dict(raw.get("params", {})).items()}
            records.append(GridRecord(run_dir=child, job_id=job_id, params=params))
            continue
        job_id, params = parsed
        records.append(GridRecord(run_dir=child, job_id=job_id, params=params))
    return records


def _extract_snr(summary: Dict[str, Any], use_physical_amplitude: bool) -> float:
    sigma_map = dict(summary.get("sigma_1d", {}))
    fid_map = dict(summary.get("fiducial", {}))
    if "A_md" not in sigma_map or "A_md" not in fid_map:
        raise KeyError("summary.json missing sigma_1d['A_md'] or fiducial['A_md']")

    sigma_a_md = float(sigma_map["A_md"])
    a_md = float(fid_map["A_md"])
    if sigma_a_md <= 0:
        raise ValueError("sigma_1d['A_md'] must be > 0")

    if not use_physical_amplitude:
        return abs(a_md) / sigma_a_md

    a_phys = float(np.exp(a_md))
    sigma_a_phys = a_phys * sigma_a_md
    return abs(a_phys) / sigma_a_phys


def build_snr_grid_table(
    runs_root: Path,
    x_param: str,
    y_param: str,
    region: str,
    dataset_sets: Sequence[str],
    manifest_path: Optional[Path] = None,
    use_physical_amplitude: bool = False,
) -> List[Dict[str, Any]]:
    records = load_grid_manifest(runs_root=runs_root, manifest_path=manifest_path)
    table: List[Dict[str, Any]] = []

    for rec in records:
        if x_param not in rec.params or y_param not in rec.params:
            continue
        for set_name in dataset_sets:
            summary_path = rec.run_dir / "fisher" / set_name / region / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            try:
                snr = _extract_snr(summary, use_physical_amplitude=use_physical_amplitude)
            except (KeyError, ValueError) as exc:
                log.warning("Skipping %s (%s): %s", summary_path, rec.job_id, exc)
                continue
            table.append(
                {
                    "job_id": rec.job_id,
                    "run_dir": str(rec.run_dir),
                    "dataset_set": set_name,
                    "region": region,
                    "x_param": x_param,
                    "y_param": y_param,
                    "x_value": float(rec.params[x_param]),
                    "y_value": float(rec.params[y_param]),
                    "snr": float(snr),
                }
            )
    if not table:
        raise RuntimeError("No matching Fisher summaries found for requested grid configuration")
    return table


def _pivot_grid(table: List[Dict[str, Any]], set_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [r for r in table if r["dataset_set"] == set_name]
    if not rows:
        raise RuntimeError(f"No rows available for dataset set '{set_name}'")
    xs = np.array(sorted({float(r["x_value"]) for r in rows}), dtype=float)
    ys = np.array(sorted({float(r["y_value"]) for r in rows}), dtype=float)
    grid = np.full((ys.size, xs.size), np.nan, dtype=float)
    x_idx = {x: i for i, x in enumerate(xs)}
    y_idx = {y: i for i, y in enumerate(ys)}
    for row in rows:
        grid[y_idx[float(row["y_value"])], x_idx[float(row["x_value"])] ] = float(row["snr"])
    return xs, ys, grid


def save_grid_outputs(
    runs_root: Path,
    x_param: str,
    y_param: str,
    region: str,
    dataset_sets: Sequence[str],
    manifest_path: Optional[Path] = None,
    use_physical_amplitude: bool = False,
    include_ratio_panel: bool = True,
    name_suffix: Optional[str] = None,
) -> Path:
    table = build_snr_grid_table(
        runs_root=runs_root,
        x_param=x_param,
        y_param=y_param,
        region=region,
        dataset_sets=dataset_sets,
        manifest_path=manifest_path,
        use_physical_amplitude=use_physical_amplitude,
    )

    out_dir = runs_root / "fisher" / "grid_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{region}_{x_param}_vs_{y_param}"
    if name_suffix:
        tag = f"{tag}_{name_suffix}"

    csv_path = out_dir / f"snr_grid_table_{tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    set_grids: Dict[str, Dict[str, np.ndarray]] = {}
    all_vals: List[float] = []
    for set_name in dataset_sets:
        xs, ys, grid = _pivot_grid(table, set_name)
        set_grids[set_name] = {"x": xs, "y": ys, "snr": grid}
        all_vals.extend(grid[np.isfinite(grid)].tolist())

    if not all_vals:
        raise RuntimeError("No finite SNR values available for plotting")

    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    panel_count = len(dataset_sets) + (1 if include_ratio_panel and len(dataset_sets) >= 2 else 0)
    fig, axes = plt.subplots(1, panel_count, figsize=(6 * panel_count, 5), squeeze=False)
    axes_flat = axes.ravel()

    for i, set_name in enumerate(dataset_sets):
        ax = axes_flat[i]
        xvals = set_grids[set_name]["x"]
        yvals = set_grids[set_name]["y"]
        grid = set_grids[set_name]["snr"]
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[xvals.min(), xvals.max(), yvals.min(), yvals.max()],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.set_title(set_name)
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="A_md SNR")

    if include_ratio_panel and len(dataset_sets) >= 2:
        ref = set_grids[dataset_sets[0]]["snr"]
        comp = set_grids[dataset_sets[1]]["snr"]
        ratio = np.divide(comp, ref, out=np.full_like(comp, np.nan), where=np.isfinite(ref) & (ref != 0))
        ax = axes_flat[-1]
        xvals = set_grids[dataset_sets[1]]["x"]
        yvals = set_grids[dataset_sets[1]]["y"]
        im = ax.imshow(
            ratio,
            origin="lower",
            aspect="auto",
            extent=[xvals.min(), xvals.max(), yvals.min(), yvals.max()],
            cmap="coolwarm",
            vmin=0.5,
            vmax=1.5,
        )
        ax.set_title(f"{dataset_sets[1]} / {dataset_sets[0]}")
        ax.set_xlabel(x_param)
        ax.set_ylabel(y_param)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="SNR ratio")
        set_grids["ratio"] = {"x": xvals, "y": yvals, "snr": ratio}

    fig.suptitle(f"Fisher A_md SNR grid: {region}")
    fig.tight_layout()
    fig_path = out_dir / f"snr_grid_{tag}.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    npz_payload = {}
    for set_name, payload in set_grids.items():
        npz_payload[f"{set_name}_x"] = payload["x"]
        npz_payload[f"{set_name}_y"] = payload["y"]
        npz_payload[f"{set_name}_snr"] = payload["snr"]
    np.savez_compressed(out_dir / f"snr_grid_arrays_{tag}.npz", **npz_payload)

    log.info("Saved Fisher grid products to %s", out_dir)
    return out_dir
