from __future__ import annotations

import copy
import csv
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    plt = None

try:
    from getdist import MCSamples
    from getdist import plots as getdist_plots
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    MCSamples = None
    getdist_plots = None

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

_DEFAULT_PARAM_TRANSFORMS = {
    "A_md": "exp",
    "chi0": "exp",
    "omega0_THz": "exp",
    "phi": "sigmoid",
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


def _filter_fitter_components_for_stokes_mode(fit_cfg: Dict[str, Any], mode: List[str]) -> None:
    wanted = {str(s).upper() for s in mode}
    fitter = fit_cfg.setdefault("fitter", {})
    components = fitter.get("components", [])

    filtered_components = []
    dropped_components = []
    for comp in components:
        comp_stokes = [str(s).upper() for s in comp.get("stokes", ["I", "Q", "U"])]
        if any(s in wanted for s in comp_stokes):
            filtered_components.append(comp)
        else:
            dropped_components.append(str(comp.get("name", "<unknown>")))

    fitter["components"] = filtered_components
    if dropped_components:
        log.info(
            "[grid] stokes %s dropping fitter components with no supported stokes output: %s",
            "".join(mode),
            ", ".join(dropped_components),
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    import numpy as np

    return 1.0 / (1.0 + np.exp(-x))


def _apply_transform(values: np.ndarray, transform_name: str) -> np.ndarray:
    import numpy as np

    name = str(transform_name).strip().lower()
    if name in {"identity", "none"}:
        return values
    if name in {"exp", "log_to_linear", "log"}:
        return np.exp(values)
    if name in {"sigmoid", "logit_to_linear", "logit"}:
        return _sigmoid(values)
    raise ValueError(f"Unsupported transform '{transform_name}'")


def _load_fisher_region_products(summary_path: Path) -> Dict[str, Any]:
    import numpy as np

    if not summary_path.exists():
        raise FileNotFoundError(f"Fisher products not found: {summary_path}")

    with np.load(summary_path, allow_pickle=False) as payload:
        names = [str(x) for x in payload["param_names"].tolist()]
        means = np.array(payload["fiducial"], dtype=float)
        cov = np.array(payload["covariance"], dtype=float)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Invalid covariance shape in {summary_path}: {cov.shape}")
    if means.shape[0] != len(names) or cov.shape[0] != len(names):
        raise ValueError(f"Inconsistent Fisher parameter dimensions in {summary_path}")

    return {"names": names, "means": means, "cov": cov}


def make_fisher_overlay_corner_plot(
    *,
    runs_root: Path,
    run_name_a: str,
    dataset_set_a: str,
    run_name_b: str,
    dataset_set_b: str,
    region: str,
    output_png: Path,
    posterior_labels: Optional[List[str]] = None,
    params: Optional[List[str]] = None,
    pretty_labels: Optional[List[str]] = None,
    param_transforms: Optional[Dict[str, str]] = None,
    true_values: Optional[Dict[str, float]] = None,
    sample_count: int = 5000,
) -> Path:
    """Build a single getdist corner plot overlaying two Fisher Gaussian posteriors.

    Parameters are loaded from fisher_products.npz for each run + dataset set.
    """

    if MCSamples is None or getdist_plots is None or plt is None:
        raise ModuleNotFoundError(
            "make_fisher_overlay_corner_plot requires optional plotting dependencies: matplotlib and getdist"
        )

    path_a = runs_root / run_name_a / "fisher" / dataset_set_a / region / "fisher_products.npz"
    path_b = runs_root / run_name_b / "fisher" / dataset_set_b / region / "fisher_products.npz"
    prod_a = _load_fisher_region_products(path_a)
    prod_b = _load_fisher_region_products(path_b)

    names_a = prod_a["names"]
    names_b = prod_b["names"]

    if params is None:
        params = [p for p in names_a if p in names_b and not p.startswith("cal_")]
    else:
        params = [str(p) for p in params]

    if not params:
        raise ValueError("No parameters available for overlay corner plot")

    missing_a = [p for p in params if p not in names_a]
    missing_b = [p for p in params if p not in names_b]
    if missing_a or missing_b:
        raise ValueError(f"Requested params missing: runA={missing_a}, runB={missing_b}")

    if pretty_labels is not None and len(pretty_labels) != len(params):
        raise ValueError("pretty_labels length must match selected params length")

    labels = [str(x) for x in pretty_labels] if pretty_labels else [str(x) for x in params]
    transforms = dict(_DEFAULT_PARAM_TRANSFORMS)
    if param_transforms:
        transforms.update({str(k): str(v) for k, v in param_transforms.items()})

    idx_a = [names_a.index(p) for p in params]
    idx_b = [names_b.index(p) for p in params]
    means_a = np.asarray(prod_a["means"][idx_a], dtype=float)
    means_b = np.asarray(prod_b["means"][idx_b], dtype=float)
    cov_a = np.asarray(prod_a["cov"][np.ix_(idx_a, idx_a)], dtype=float)
    cov_b = np.asarray(prod_b["cov"][np.ix_(idx_b, idx_b)], dtype=float)

    rng = np.random.default_rng(0)
    draws_a = rng.multivariate_normal(means_a, cov_a, size=int(sample_count))
    draws_b = rng.multivariate_normal(means_b, cov_b, size=int(sample_count))

    for i, pname in enumerate(params):
        transform_name = transforms.get(pname, "identity")
        draws_a[:, i] = _apply_transform(draws_a[:, i], transform_name)
        draws_b[:, i] = _apply_transform(draws_b[:, i], transform_name)

    sample_label_a = posterior_labels[0] if posterior_labels and len(posterior_labels) > 0 else f"{dataset_set_a}:{run_name_a}"
    sample_label_b = posterior_labels[1] if posterior_labels and len(posterior_labels) > 1 else f"{dataset_set_b}:{run_name_b}"

    mcs_a = MCSamples(samples=draws_a, names=params, labels=labels, label=sample_label_a)
    mcs_b = MCSamples(samples=draws_b, names=params, labels=labels, label=sample_label_b)

    markers: Dict[str, float] = {}
    if true_values:
        for pname in params:
            if pname in true_values:
                markers[pname] = float(true_values[pname])
    else:
        for i, pname in enumerate(params):
            markers[pname] = float(_apply_transform(np.array([means_a[i]], dtype=float), transforms.get(pname, "identity"))[0])

    plotter = getdist_plots.get_subplot_plotter()
    plotter.triangle_plot(
        [mcs_a, mcs_b],
        params=params,
        filled=True,
        legend_labels=[sample_label_a, sample_label_b],
        markers=markers,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plotter.fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(plotter.fig)
    log.info("Saved Fisher overlay corner plot to %s", output_png)
    return output_png


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
    reuse_simulation_h5: Optional[Path] = None,
    skip_simulations: bool = False,
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
        grid_cfg=grid,
        model_cfg=dict(cfg.get("model", {})),
        gain_error_groups=gain_error_groups,
        reuse_simulation_h5=reuse_simulation_h5 or workflow.get("reuse_simulation_h5"),
        skip_simulations=skip_simulations or bool(workflow.get("skip_simulations", False)),
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
    grid_cfg: Dict[str, Any],
    model_cfg: Optional[Dict[str, Any]] = None,
    gain_error_groups: Optional[Dict[str, Any]] = None,
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
    grid_cfg = dict(grid_cfg)
    grid_tag = str(grid_cfg["grid_tag"])
    x_param = str(grid_cfg["x_param"])
    y_param = str(grid_cfg["y_param"])
    region = str(grid_cfg["region"])
    dataset_sets = [str(x) for x in grid_cfg.get("dataset_sets", ["baseline", "baseline_plus_litebird"])]
    use_physical_amplitude = bool(grid_cfg.get("use_physical_amplitude", False))
    include_ratio_panel = bool(grid_cfg.get("include_ratio_panel", True))
    stokes_modes = _normalize_stokes_modes(grid_cfg.get("stokes_modes"))
    grid_parameters = list(grid_cfg.get("parameters", []))

    shared_sims_h5 = Path(reuse_simulation_h5) if reuse_simulation_h5 is not None else None
    should_run_simulations = not skip_simulations and shared_sims_h5 is None
    parameter_lookup = {str(p.get("name")): p for p in grid_parameters}

    runs_root = Path(out_dir) / grid_tag
    runs_root.mkdir(parents=True, exist_ok=True)
    tmp_cfg_dir = runs_root / "_grid_tmp_configs"
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)

    resolved_manifest_jobs_by_mode: Dict[str, Dict[str, Dict[str, Any]]] = {
        _stokes_mode_tag(mode): {} for mode in stokes_modes
    }

    for idx, job in enumerate(jobs):
        job_id = str(job.get("job_id", f"j{idx:04d}"))
        base_run_name = str(job.get("run_name", job_id))
        sim_overrides = dict(job.get("sim_overrides", {}))
        fitter_overrides = dict(job.get("fitter_overrides", {}))
        params = {str(k): float(v) for k, v in dict(job.get("params", {})).items()}

        sim_cfg_obj = _deep_update(base_sim_cfg, sim_overrides)
        fit_cfg_obj = _deep_update(base_fit_cfg, fitter_overrides)

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

        sim_out_h5 = shared_sims_h5 or (runs_root / base_run_name / "products" / "simulations" / "simulations.h5")
        if should_run_simulations:
            sim_out_h5.parent.mkdir(parents=True, exist_ok=True)

        sim_cfg_obj.setdefault("simulations", {})["out_h5"] = str(sim_out_h5)
        fit_cfg_obj.setdefault("fitter", {})["sims_h5"] = str(sim_out_h5)

        sim_cfg_path = tmp_cfg_dir / f"{job_id}_sim.yaml"
        sim_cfg_path.write_text(yaml.safe_dump(sim_cfg_obj, sort_keys=False))

        if should_run_simulations:
            log.info("[grid] running simulation for %s", job_id)
            run_simulations(sim_cfg_path, overwrite=overwrite, dry_run=dry_run)

        for mode in stokes_modes:
            mode_tag = _stokes_mode_tag(mode)
            run_name = f"{base_run_name}_mode{mode_tag}"
            mode_fit_cfg_obj = copy.deepcopy(fit_cfg_obj)
            _filter_fitter_components_for_stokes_mode(mode_fit_cfg_obj, mode)
            mode_fit_cfg_obj.setdefault("fitter", {})["stokes_fit"] = list(mode)
            mode_fit_cfg_obj.setdefault("fitter", {})["out_dir"] = str(out_dir)
            mode_fit_cfg_obj.setdefault("fitter", {})["sims_tag"] = grid_tag

            fit_cfg_path = tmp_cfg_dir / f"{job_id}_fit_mode{mode_tag}.yaml"
            fit_cfg_path.write_text(yaml.safe_dump(mode_fit_cfg_obj, sort_keys=False))

            log.info("[grid] running fisher for %s (%s)", job_id, mode_tag)
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

            resolved_manifest_jobs_by_mode[mode_tag][job_id] = {
                "run_name": run_name,
                "params": params,
                "stokes_mode": list(mode),
                "stokes_mode_tag": mode_tag,
            }

    fisher_dir = runs_root / "fisher"
    fisher_dir.mkdir(parents=True, exist_ok=True)
    for mode in stokes_modes:
        mode_tag = _stokes_mode_tag(mode)
        manifest_for_plots = fisher_dir / f"grid_manifest_resolved_mode{mode_tag}.json"
        manifest_for_plots.write_text(json.dumps({"mode_tag": mode_tag, "jobs": resolved_manifest_jobs_by_mode[mode_tag]}, indent=2))

        save_grid_outputs(
            runs_root=runs_root,
            x_param=x_param,
            y_param=y_param,
            region=region,
            dataset_sets=dataset_sets,
            manifest_path=manifest_for_plots,
            use_physical_amplitude=use_physical_amplitude,
            include_ratio_panel=include_ratio_panel,
            name_suffix=f"mode{mode_tag}",
        )

    comparison = _build_snr_mode_comparison(
        runs_root=runs_root,
        mode_manifests=resolved_manifest_jobs_by_mode,
        region=region,
        use_physical_amplitude=use_physical_amplitude,
    )

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
    import numpy as np
