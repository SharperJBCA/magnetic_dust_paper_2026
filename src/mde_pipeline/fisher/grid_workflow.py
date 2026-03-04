from __future__ import annotations

import copy
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..fisher.grid_plots import save_grid_outputs
from ..fisher.run import run_fisher
from ..simulations.run import run_simulations
from ..utils.config import load_yaml
from ..utils.logging import get_logger

log = get_logger(__name__)


_GAIN_GROUP_PREFIXES = {
    "planck": "planck",
    "wmap": "wmap",
    "litebird": "litebird",
    "cbass": "cbass",
}

_DEFAULT_STOKES_MODES = [["I", "Q", "U"], ["Q", "U"]]


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _normalize_jobs(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    jobs = manifest.get("jobs", [])
    if isinstance(jobs, dict):
        return [{"job_id": jid, **meta} for jid, meta in jobs.items()]
    if isinstance(jobs, list):
        return [dict(j) for j in jobs]
    raise ValueError("manifest 'jobs' must be a list or dict")


def _set_sim_component_param(sim_cfg: Dict[str, Any], comp_name: str, param_name: str, value: float) -> None:
    for comp in sim_cfg.setdefault("simulations", {}).get("components", []):
        if str(comp.get("name")) == comp_name:
            comp.setdefault("params", {})[param_name] = float(value)
            return
    raise ValueError(f"Simulation component '{comp_name}' not found")


def _set_fitter_param(fitter_cfg: Dict[str, Any], param_name: str, value: float) -> None:
    for comp in fitter_cfg.setdefault("fitter", {}).get("components", []):
        fixed = comp.setdefault("fixed_params", {})
        init = comp.get("init")
        params_map = dict(comp.get("params_map", {}))

        if param_name in fixed:
            fixed[param_name] = float(value)
        if isinstance(init, dict) and param_name in init:
            init[param_name] = [float(value), 0.0]
        if param_name in params_map.values():
            fixed[param_name] = float(value)


def _filter_component_lists(
    sim_cfg: Dict[str, Any],
    fit_cfg: Dict[str, Any],
    keep_sim_components: Optional[List[str]],
    keep_fitter_components: Optional[List[str]],
) -> None:
    aliases = {
        "thermaldust": "dust",
    }

    def _canonical_name(name: str) -> str:
        lowered = name.lower()
        return aliases.get(lowered, lowered)

    if keep_sim_components:
        allowed = {_canonical_name(str(x)) for x in keep_sim_components}
        sim_cfg["simulations"]["components"] = [
            c
            for c in sim_cfg.setdefault("simulations", {}).get("components", [])
            if _canonical_name(str(c.get("name"))) in allowed
        ]
    if keep_fitter_components:
        allowed = {_canonical_name(str(x)) for x in keep_fitter_components}
        fit_cfg["fitter"]["components"] = [
            c
            for c in fit_cfg.setdefault("fitter", {}).get("components", [])
            if _canonical_name(str(c.get("name"))) in allowed
        ]


def _apply_gain_error_group_overrides(fitter_cfg: Dict[str, Any], gain_error_groups: Optional[Dict[str, Any]]) -> None:
    if not gain_error_groups:
        return

    gains = fitter_cfg.setdefault("fitter", {}).get("gains", [])
    if not gains:
        return

    normalized_groups: Dict[str, float] = {}
    for group_name, sigma in gain_error_groups.items():
        key = str(group_name).lower()
        if key in _GAIN_GROUP_PREFIXES:
            normalized_groups[key] = float(sigma)

    for gain in gains:
        gain_name = str(gain.get("name", "")).lower()
        matching_sigma = None
        for group_key, prefix in _GAIN_GROUP_PREFIXES.items():
            if gain_name.startswith(prefix) and group_key in normalized_groups:
                matching_sigma = normalized_groups[group_key]
                break

        if matching_sigma is None:
            continue

        for prior in gain.get("priors", []):
            if str(prior.get("type", "")).lower() != "normal":
                continue
            params = prior.get("params", {})
            for param_values in params.values():
                if isinstance(param_values, list) and len(param_values) >= 2:
                    param_values[1] = matching_sigma


def _build_jobs_from_grid_section(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    grid = dict(cfg.get("grid", {}))
    parameters = list(grid.get("parameters", []))
    names = [str(p["name"]) for p in parameters]
    values = [list(p.get("values", [])) for p in parameters]
    if len(parameters) < 2:
        raise ValueError("grid.parameters must include at least 2 parameters")
    if any(len(v) == 0 for v in values):
        raise ValueError("Each grid parameter needs at least one value")

    jobs: List[Dict[str, Any]] = []
    for idx, combo in enumerate(product(*values)):
        pvals = {k: float(v) for k, v in zip(names, combo)}
        run_name = f"j{idx:04d}_" + "_".join(f"{k}{v:g}" for k, v in pvals.items())
        jobs.append({"job_id": f"j{idx:04d}", "run_name": run_name, "params": pvals})
    return jobs


def _normalize_stokes_modes(raw_modes: Optional[List[Any]]) -> List[List[str]]:
    if raw_modes is None:
        return [list(mode) for mode in _DEFAULT_STOKES_MODES]

    normalized: List[List[str]] = []
    for item in raw_modes:
        if isinstance(item, str):
            mode = [s.strip().upper() for s in item.split(",") if s.strip()]
        else:
            mode = [str(s).upper() for s in item]

        if not mode:
            raise ValueError("Each stokes mode must contain at least one stokes component")

        invalid = [s for s in mode if s not in {"I", "Q", "U"}]
        if invalid:
            raise ValueError(f"Invalid stokes mode entries: {invalid}")

        if mode not in normalized:
            normalized.append(mode)

    if not normalized:
        raise ValueError("At least one stokes mode is required")
    return normalized


def _stokes_mode_tag(mode: List[str]) -> str:
    return "".join(mode)


def _build_snr_mode_comparison(
    *,
    runs_root: Path,
    mode_manifests: Dict[str, Dict[str, Dict[str, Any]]],
    region: str,
    use_physical_amplitude: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"region": region, "modes": {}}
    for mode_tag, jobs in mode_manifests.items():
        mode_rows: Dict[str, Any] = {}
        for job_id, meta in jobs.items():
            run_name = str(meta["run_name"])
            params = dict(meta.get("params", {}))
            base_summary_path = runs_root / run_name / "fisher" / "baseline" / region / "summary.json"
            plus_summary_path = runs_root / run_name / "fisher" / "baseline_plus_litebird" / region / "summary.json"
            if not base_summary_path.exists() or not plus_summary_path.exists():
                continue

            base_summary = json.loads(base_summary_path.read_text())
            plus_summary = json.loads(plus_summary_path.read_text())

            sigma_map_base = dict(base_summary.get("sigma_1d", {}))
            fid_map_base = dict(base_summary.get("fiducial", {}))
            sigma_map_plus = dict(plus_summary.get("sigma_1d", {}))
            fid_map_plus = dict(plus_summary.get("fiducial", {}))
            if "A_md" not in sigma_map_base or "A_md" not in fid_map_base:
                continue
            if "A_md" not in sigma_map_plus or "A_md" not in fid_map_plus:
                continue

            sigma_base = float(sigma_map_base["A_md"])
            sigma_plus = float(sigma_map_plus["A_md"])
            if sigma_base <= 0 or sigma_plus <= 0:
                continue

            if use_physical_amplitude:
                snr_base = 1.0 / sigma_base
                snr_plus = 1.0 / sigma_plus
            else:
                snr_base = abs(float(fid_map_base["A_md"])) / sigma_base
                snr_plus = abs(float(fid_map_plus["A_md"])) / sigma_plus
            snr_ratio = None if snr_base == 0 else (snr_plus / snr_base)
            mode_rows[job_id] = {
                "run_name": run_name,
                "params": params,
                "snr_baseline": snr_base,
                "snr_plus": snr_plus,
                "snr_abs_diff": snr_plus - snr_base,
                "snr_ratio": snr_ratio,
            }

        out["modes"][mode_tag] = mode_rows
    return out


def run_fisher_grid_from_yaml(
    grid_yaml: Path,
    tag: str,
    out_dir: Optional[Path] = None,
    regions_h5: Optional[Path] = None,
    processed_h5: Optional[Path] = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Path:
    cfg = load_yaml(grid_yaml)
    workflow = dict(cfg.get("workflow", {}))
    grid = dict(cfg.get("grid", {}))
    gain_error_groups = dict(cfg.get("model", {}).get("gain_error_groups", {}))
    gain_error_groups = _deep_update(gain_error_groups, dict(grid.get("gain_error_groups", {})))

    jobs = _normalize_jobs(cfg) if cfg.get("jobs") is not None else _build_jobs_from_grid_section(cfg)
    if not jobs:
        raise ValueError("No jobs found in grid yaml")

    tmp_manifest = Path(workflow.get("manifest_out", "products/fits/grid_manifest_runtime.json"))
    tmp_manifest.parent.mkdir(parents=True, exist_ok=True)
    tmp_manifest.write_text(json.dumps({"jobs": jobs}, indent=2))

    return run_fisher_grid_workflow(
        grid_manifest=tmp_manifest,
        sims_cfg=Path(workflow["sims_config"]),
        fitter_cfg=Path(workflow["fisher_config"]),
        data_yaml=Path(workflow["data_config"]),
        templates_yaml=Path(workflow["templates_config"]),
        regions_h5=Path(regions_h5 or workflow.get("regions_h5", f"products/regions/{tag}/regions.h5")),
        processed_h5=Path(processed_h5 or workflow.get("processed_h5", f"products/processed_maps/{tag}/processed_maps.h5")),
        out_dir=Path(out_dir or workflow.get("out_dir", f"products/fits/{tag}")),
        grid_tag=str(grid["grid_tag"]),
        x_param=str(grid["x_param"]),
        y_param=str(grid["y_param"]),
        region=str(grid["region"]),
        dataset_sets=[str(x) for x in grid.get("dataset_sets", ["baseline", "baseline_plus_litebird"])],
        use_physical_amplitude=bool(grid.get("use_physical_amplitude", False)),
        include_ratio_panel=bool(grid.get("include_ratio_panel", True)),
        stokes_modes=_normalize_stokes_modes(grid.get("stokes_modes")),
        model_cfg=dict(cfg.get("model", {})),
        gain_error_groups=gain_error_groups,
        grid_parameters=list(grid.get("parameters", [])),
        reuse_simulation_h5=workflow.get("reuse_simulation_h5"),
        skip_simulations=bool(workflow.get("skip_simulations", False)),
        overwrite=overwrite,
        dry_run=dry_run,
    )


def run_fisher_grid_workflow(
    grid_manifest: Path,
    sims_cfg: Path,
    fitter_cfg: Path,
    data_yaml: Path,
    templates_yaml: Path,
    regions_h5: Path,
    processed_h5: Path,
    out_dir: Path,
    grid_tag: str,
    x_param: str,
    y_param: str,
    region: str,
    dataset_sets: List[str],
    use_physical_amplitude: bool = False,
    include_ratio_panel: bool = True,
    stokes_modes: Optional[List[List[str]]] = None,
    model_cfg: Optional[Dict[str, Any]] = None,
    gain_error_groups: Optional[Dict[str, Any]] = None,
    grid_parameters: Optional[List[Dict[str, Any]]] = None,
    reuse_simulation_h5: Optional[Path] = None,
    skip_simulations: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Path:
    manifest = json.loads(grid_manifest.read_text())
    jobs = _normalize_jobs(manifest)
    if not jobs:
        raise ValueError("grid manifest has no jobs")

    base_sim_cfg = load_yaml(sims_cfg)
    base_fit_cfg = load_yaml(fitter_cfg)
    model_cfg = model_cfg or {}
    shared_sims_h5 = Path(reuse_simulation_h5) if reuse_simulation_h5 is not None else None
    should_run_simulations = not skip_simulations and shared_sims_h5 is None
    parameter_lookup = {str(p.get("name")): p for p in (grid_parameters or [])}
    stokes_modes = _normalize_stokes_modes(stokes_modes)

    runs_root = Path(out_dir) / grid_tag
    runs_root.mkdir(parents=True, exist_ok=True)
    tmp_cfg_dir = runs_root / "_grid_tmp_configs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)

    resolved_manifest_jobs_by_mode: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for mode in stokes_modes:
        mode_tag = _stokes_mode_tag(mode)
        resolved_manifest_jobs: Dict[str, Dict[str, Any]] = {}

        for idx, job in enumerate(jobs):
            job_id = str(job.get("job_id", f"j{idx:04d}"))
            run_name = str(job.get("run_name", job_id))
            mode_run_name = f"{run_name}__stokes_{mode_tag}"
            sim_overrides = dict(job.get("sim_overrides", {}))
            fitter_overrides = dict(job.get("fitter_overrides", {}))
            params = {str(k): float(v) for k, v in dict(job.get("params", {})).items()}

            sim_cfg_obj = _deep_update(base_sim_cfg, sim_overrides)
            fit_cfg_obj = _deep_update(base_fit_cfg, fitter_overrides)

            _apply_gain_error_group_overrides(fit_cfg_obj, gain_error_groups)

            _filter_component_lists(
                sim_cfg_obj,
                fit_cfg_obj,
                keep_sim_components=[str(x) for x in model_cfg.get("simulation_components", [])],
                keep_fitter_components=[str(x) for x in model_cfg.get("fitter_components", [])],
            )

            for comp_name, cmap in dict(model_cfg.get("fixed_simulation_params", {})).items():
                for p, v in dict(cmap).items():
                    _set_sim_component_param(sim_cfg_obj, str(comp_name), str(p), float(v))
            for p, v in dict(model_cfg.get("fixed_fitter_params", {})).items():
                _set_fitter_param(fit_cfg_obj, str(p), float(v))

            for p_name, p_val in params.items():
                pmeta = parameter_lookup.get(p_name, {})
                sim_comp = pmeta.get("sim_component")
                sim_param = pmeta.get("sim_param")
                fit_param = pmeta.get("fitter_param")
                if sim_comp and sim_param:
                    _set_sim_component_param(sim_cfg_obj, str(sim_comp), str(sim_param), float(p_val))
                if fit_param:
                    _set_fitter_param(fit_cfg_obj, str(fit_param), float(p_val))

            sim_out_h5 = shared_sims_h5 or (runs_root / run_name / "products" / "simulations" / "simulations.h5")
            if should_run_simulations:
                sim_out_h5.parent.mkdir(parents=True, exist_ok=True)

            sim_cfg_obj.setdefault("simulations", {})["out_h5"] = str(sim_out_h5)
            fit_cfg_obj.setdefault("fitter", {})["sims_h5"] = str(sim_out_h5)
            fit_cfg_obj.setdefault("fitter", {})["out_dir"] = str(out_dir)
            fit_cfg_obj.setdefault("fitter", {})["sims_tag"] = grid_tag
            fit_cfg_obj.setdefault("fitter", {})["stokes_fit"] = list(mode)

            sim_cfg_path = tmp_cfg_dir / f"{job_id}_{mode_tag}_sim.yaml"
            fit_cfg_path = tmp_cfg_dir / f"{job_id}_{mode_tag}_fit.yaml"
            sim_cfg_path.write_text(yaml.safe_dump(sim_cfg_obj, sort_keys=False))
            fit_cfg_path.write_text(yaml.safe_dump(fit_cfg_obj, sort_keys=False))

            if should_run_simulations:
                log.info("[grid] running simulation for %s (%s)", job_id, mode_tag)
                run_simulations(sim_cfg_path, overwrite=overwrite, dry_run=dry_run)

            log.info("[grid] running fisher for %s (%s)", job_id, mode_tag)
            run_fisher(
                fitter_yaml=fit_cfg_path,
                data_yaml=data_yaml,
                templates_yaml=templates_yaml,
                regions_h5=regions_h5,
                processed_h5=processed_h5,
                out_dir=out_dir,
                run_name=mode_run_name,
                region_ids=[region],
                overwrite=overwrite,
                dry_run=dry_run,
            )

            resolved_manifest_jobs[job_id] = {"run_name": mode_run_name, "params": params}

        resolved_manifest_jobs_by_mode[mode_tag] = resolved_manifest_jobs

        resolved_manifest_path = runs_root / f"grid_manifest_resolved_{mode_tag}.json"
        resolved_manifest_path.write_text(json.dumps({"jobs": resolved_manifest_jobs}, indent=2))

        save_grid_outputs(
            runs_root=runs_root,
            x_param=x_param,
            y_param=y_param,
            region=region,
            dataset_sets=dataset_sets,
            manifest_path=resolved_manifest_path,
            use_physical_amplitude=use_physical_amplitude,
            include_ratio_panel=include_ratio_panel,
            name_suffix=f"stokes_{mode_tag}",
        )

    comparison = _build_snr_mode_comparison(
        runs_root=runs_root,
        mode_manifests=resolved_manifest_jobs_by_mode,
        region=region,
        use_physical_amplitude=use_physical_amplitude,
    )

    fisher_dir = runs_root / "fisher"
    fisher_dir.mkdir(parents=True, exist_ok=True)
    (fisher_dir / "snr_mode_comparison.json").write_text(json.dumps(comparison, indent=2))

    grid_maps_dir = fisher_dir / "grid_maps"
    grid_maps_dir.mkdir(parents=True, exist_ok=True)
    csv_path = grid_maps_dir / "snr_mode_comparison_table.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "region",
                "stokes_mode",
                "job_id",
                "run_name",
                "snr_baseline",
                "snr_plus",
                "snr_abs_diff",
                "snr_ratio",
                "params_json",
            ],
        )
        writer.writeheader()
        for mode_tag, rows in comparison.get("modes", {}).items():
            for job_id, row in rows.items():
                writer.writerow(
                    {
                        "region": region,
                        "stokes_mode": mode_tag,
                        "job_id": job_id,
                        "run_name": row["run_name"],
                        "snr_baseline": row["snr_baseline"],
                        "snr_plus": row["snr_plus"],
                        "snr_abs_diff": row["snr_abs_diff"],
                        "snr_ratio": row["snr_ratio"],
                        "params_json": json.dumps(row.get("params", {}), sort_keys=True),
                    }
                )

    return grid_maps_dir
