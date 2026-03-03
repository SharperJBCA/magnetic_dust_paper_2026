from __future__ import annotations

import copy
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

    runs_root = Path(out_dir) / grid_tag
    runs_root.mkdir(parents=True, exist_ok=True)
    tmp_cfg_dir = runs_root / "_grid_tmp_configs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)

    resolved_manifest_jobs: Dict[str, Dict[str, Any]] = {}

    for idx, job in enumerate(jobs):
        job_id = str(job.get("job_id", f"j{idx:04d}"))
        run_name = str(job.get("run_name", job_id))
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

        sim_cfg_path = tmp_cfg_dir / f"{job_id}_sim.yaml"
        fit_cfg_path = tmp_cfg_dir / f"{job_id}_fit.yaml"
        sim_cfg_path.write_text(yaml.safe_dump(sim_cfg_obj, sort_keys=False))
        fit_cfg_path.write_text(yaml.safe_dump(fit_cfg_obj, sort_keys=False))

        if should_run_simulations:
            log.info("[grid] running simulation for %s", job_id)
            run_simulations(sim_cfg_path, overwrite=overwrite, dry_run=dry_run)

        log.info("[grid] running fisher for %s", job_id)
        run_fisher(
            fitter_yaml=fit_cfg_path,
            data_yaml=data_yaml,
            templates_yaml=templates_yaml,
            regions_h5=regions_h5,
            processed_h5=processed_h5,
            out_dir=out_dir,
            run_name=run_name,
            region_ids=[region],
            overwrite=overwrite,
            dry_run=dry_run,
        )

        resolved_manifest_jobs[job_id] = {"run_name": run_name, "params": params}

    resolved_manifest_path = runs_root / "grid_manifest_resolved.json"
    resolved_manifest_path.write_text(json.dumps({"jobs": resolved_manifest_jobs}, indent=2))

    return save_grid_outputs(
        runs_root=runs_root,
        x_param=x_param,
        y_param=y_param,
        region=region,
        dataset_sets=dataset_sets,
        manifest_path=resolved_manifest_path,
        use_physical_amplitude=use_physical_amplitude,
        include_ratio_panel=include_ratio_panel,
    )
