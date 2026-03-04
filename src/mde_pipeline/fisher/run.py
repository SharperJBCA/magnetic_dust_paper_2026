from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from getdist import MCSamples
from getdist import plots as getdist_plots

from ..fitting.data_types import FitData, Model, build_components_from_yaml
from ..fitting.fisher import fisher_gain_marginalized
from ..fitting.run import _resolve_fit_stokes, _resolve_target_entries
from ..io.maps_io import MapIO
from ..io.regions_io import RegionsIO
from ..templates.templates import load_templates_config
from ..utils.config import load_yaml
from ..utils.logging import get_logger
from ..fitting.priors import BoundsPrior

log = get_logger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _collect_dataset_sets(fitter_info: Dict[str, Any]) -> Dict[str, List[str]]:
    dataset_sets = fitter_info.get("dataset_sets", {})
    if dataset_sets:
        return {k: list(v) for k, v in dataset_sets.items()}

    targets = fitter_info.get("targets", [])
    if not targets:
        raise ValueError("fitter.targets (or fitter.dataset_sets) must define at least one target.")
    return {"default": list(targets)}


def _fisher_inversion_config(fitter_info: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(fitter_info.get("fisher_inversion", {}))
    return {
        "scaling": str(cfg.get("scaling", "diag")).strip().lower(),
        "diag_floor": float(cfg.get("diag_floor", 1e-12)),
        "fiducial_floor": float(cfg.get("fiducial_floor", 1e-6)),
        "ridge": float(cfg.get("ridge", 0.0)),
    }


def _fisher_scale_vector(
    F: np.ndarray,
    param_names: List[str],
    params0: Dict[str, float],
    *,
    method: str,
    diag_floor: float,
    fiducial_floor: float,
) -> np.ndarray:
    if method in {"none", "identity"}:
        return np.ones(F.shape[0], dtype=float)

    if method in {"diag", "fisher_diag"}:
        diag = np.clip(np.diag(F), diag_floor, np.inf)
        return 1.0 / np.sqrt(diag)

    if method in {"fiducial", "param"}:
        scales = []
        for name in param_names:
            scales.append(max(abs(float(params0[name])), fiducial_floor))
        return np.asarray(scales, dtype=float)

    raise ValueError(f"Unsupported fisher inversion scaling '{method}'")


def _invert_fisher_stably(
    F: np.ndarray,
    param_names: List[str],
    params0: Dict[str, float],
    inversion_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, float]]:
    scaling = str(inversion_cfg.get("scaling", "diag")).lower()
    diag_floor = float(inversion_cfg.get("diag_floor", 1e-12))
    fiducial_floor = float(inversion_cfg.get("fiducial_floor", 1e-6))
    ridge = float(inversion_cfg.get("ridge", 0.0))

    scale = _fisher_scale_vector(
        F,
        param_names,
        params0,
        method=scaling,
        diag_floor=diag_floor,
        fiducial_floor=fiducial_floor,
    )
    S = np.diag(scale)
    F_scaled = S @ F @ S
    if ridge > 0:
        F_scaled = F_scaled + ridge * np.eye(F_scaled.shape[0], dtype=float)

    cov_scaled = np.linalg.pinv(F_scaled)
    cov = S @ cov_scaled @ S

    with np.errstate(invalid="ignore", divide="ignore"):
        cond_raw = float(np.linalg.cond(F))
        cond_scaled = float(np.linalg.cond(F_scaled))

    diagnostics = {
        "conditioning_raw": cond_raw,
        "conditioning_scaled": cond_scaled,
        "scaling": scaling,
        "diag_floor": diag_floor,
        "fiducial_floor": fiducial_floor,
        "ridge": ridge,
    }
    return cov, diagnostics


def _magnetic_dust_params(param_names: List[str]) -> List[str]:
    wanted = {"a_md", "chi0", "phi", "phi_md", "omega0_thz"}
    out: List[str] = []
    for p in param_names:
        pl = p.lower()
        if ("md" in pl) or (pl in wanted):
            out.append(p)
    return out


def _snr_summary(params0: Dict[str, float], sigma_by_param: Dict[str, float], md_params: List[str]) -> Dict[str, Optional[float]]:
    summary: Dict[str, Optional[float]] = {}
    for name in md_params:
        sigma = sigma_by_param.get(name)
        if sigma is None or sigma <= 0:
            summary[name] = None
            continue
        summary[name] = float(abs(params0[name]) / sigma)
    return summary


def _write_fisher_region_products(
    reg_out: Path,
    fisher: np.ndarray,
    cov: np.ndarray,
    param_names: List[str],
    params0: Dict[str, float],
    sigma_map: Dict[str, float],
    include_gain_params_in_corner: bool,
) -> None:
    np.savez_compressed(
        reg_out / "fisher_products.npz",
        fisher=fisher,
        covariance=cov,
        param_names=np.array(param_names, dtype="U"),
        fiducial=np.array([float(params0[p]) for p in param_names], dtype=float),
        sigma_1d=np.array([float(sigma_map[p]) for p in param_names], dtype=float),
    )

    corner_names = list(param_names)
    if not include_gain_params_in_corner:
        corner_names = [p for p in corner_names if not p.startswith("cal_")]
    if not corner_names:
        return

    means = np.array([float(params0[p]) for p in corner_names], dtype=float)
    idx = [param_names.index(p) for p in corner_names]
    cov_plot_base = np.asarray(cov[np.ix_(idx, idx)], dtype=float)
    draw_count = 3000
    rng = np.random.default_rng(0)

    sample_draws: Optional[np.ndarray] = None
    jitter_levels = [0.0, 1e-12, 1e-9, 1e-6]
    for jitter in jitter_levels:
        try:
            cov_plot = cov_plot_base.copy()
            if jitter > 0:
                cov_plot = cov_plot + jitter * np.eye(cov_plot.shape[0], dtype=float)
            sample_draws = rng.multivariate_normal(means, cov_plot, size=draw_count, check_valid="raise")
            break
        except (np.linalg.LinAlgError, ValueError):
            try:
                fisher_plot = np.linalg.pinv(cov_plot_base)
                cov_plot_base = np.linalg.pinv(
                    fisher_plot + max(jitter, 1e-12) * np.eye(fisher_plot.shape[0], dtype=float)
                )
            except np.linalg.LinAlgError:
                continue

    if sample_draws is None:
        log.warning("Skipping Fisher corner plot: could not draw samples from covariance.")
        return

    samples = MCSamples(samples=sample_draws, names=corner_names, labels=corner_names)
    plotter = getdist_plots.get_subplot_plotter()
    plotter.triangle_plot([samples], params=corner_names, filled=True)
    plotter.fig.savefig(reg_out / "corner.png", dpi=180)
    plt.close(plotter.fig)


def _marker_for_map(map_name: str) -> str:
    name = map_name.lower()
    if "cbass" in name or "spass" in name:
        return "o"
    if "wmap" in name:
        return "s"
    if "planck" in name:
        return "^"
    if "litebird" in name:
        return "D"
    return "x"


def _eval_component_region_mean(spec, comp, template, nu_ghz: float, params: Dict[str, float]) -> Tuple[float, float]:
    comp_params = dict(spec.fixed_params)
    comp_params.update({arg: params[name] for arg, name in spec.params_map.items() if name in params})
    out = comp.evaluate(nu_ghz=nu_ghz, T=template, params=comp_params)
    z = np.zeros_like(template.I)
    i_val = float(np.mean(out.get("I", z)))
    q_val = float(np.mean(out.get("Q", z)))
    u_val = float(np.mean(out.get("U", z)))
    p_val = float(np.sqrt(q_val * q_val + u_val * u_val))
    return i_val, p_val


def _collect_parameter_bounds(global_prior: Any) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    def _add_bound(name: str, lo: Optional[float], hi: Optional[float]) -> None:
        prev = bounds.get(name)
        if prev is None:
            bounds[name] = (lo, hi)
            return
        prev_lo, prev_hi = prev
        next_lo = lo if prev_lo is None else (max(prev_lo, lo) if lo is not None else prev_lo)
        next_hi = hi if prev_hi is None else (min(prev_hi, hi) if hi is not None else prev_hi)
        bounds[name] = (next_lo, next_hi)

    if global_prior is None:
        return bounds

    if hasattr(global_prior, "bounds") and isinstance(global_prior.bounds, dict):
        for p, v in global_prior.bounds.items():
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                _add_bound(str(p), v[0], v[1])

    if hasattr(global_prior, "priors"):
        for pr in getattr(global_prior, "priors"):
            if isinstance(pr, BoundsPrior) and hasattr(pr, "bounds"):
                for p, v in pr.bounds.items():
                    if isinstance(v, (list, tuple)) and len(v) >= 2:
                        _add_bound(str(p), v[0], v[1])

    return bounds


def _draw_valid_posterior_samples(
    means: np.ndarray,
    cov: np.ndarray,
    param_names: List[str],
    bounds_lookup: Dict[str, Tuple[Optional[float], Optional[float]]],
    sample_count: int,
    rng: np.random.Generator,
    max_attempt_factor: int = 30,
) -> np.ndarray:
    max_attempts = max(sample_count * max_attempt_factor, sample_count)
    accepted: List[np.ndarray] = []
    attempts = 0

    while len(accepted) < sample_count and attempts < max_attempts:
        draw = rng.multivariate_normal(means, cov)
        attempts += 1
        is_valid = True
        for idx, pname in enumerate(param_names):
            if pname not in bounds_lookup:
                continue
            lo, hi = bounds_lookup[pname]
            value = float(draw[idx])
            if lo is not None and value < lo:
                is_valid = False
                break
            if hi is not None and value > hi:
                is_valid = False
                break
        if is_valid:
            accepted.append(draw)

    if not accepted:
        return np.empty((0, len(param_names)), dtype=float)
    return np.asarray(accepted, dtype=float)


def _write_fisher_sed_plots(
    reg_out: Path,
    model: Model,
    fitdata: FitData,
    params0: Dict[str, float],
    cov: np.ndarray,
    param_names: List[str],
    global_prior: Any = None,
) -> None:
    nu_plot = np.logspace(np.log10(max(1.0, fitdata.frequencies_ghz.min() * 0.8)), np.log10(fitdata.frequencies_ghz.max() * 1.2), 220)
    rng = np.random.default_rng(0)
    means = np.array([params0[p] for p in param_names], dtype=float)
    bounds_lookup = _collect_parameter_bounds(global_prior)
    try:
        draws = _draw_valid_posterior_samples(
            means=means,
            cov=cov,
            param_names=param_names,
            bounds_lookup=bounds_lookup,
            sample_count=140,
            rng=rng,
        )
    except np.linalg.LinAlgError:
        return
    if draws.shape[0] == 0:
        return
    draw_params = [{n: float(v) for n, v in zip(param_names, draws[j])} for j in range(draws.shape[0])]

    comp_labels = [spec.name for spec, _comp in model._comps]
    comp_colors = {label: f"C{i % 10}" for i, label in enumerate(comp_labels)}

    marker_by_freq = {float(nu): _marker_for_map(mn) for nu, mn in zip(fitdata.frequencies_ghz, fitdata.map_names)}

    for mode, ylabel in [("I", "Intensity (region mean)"), ("P", "Polarized intensity (region mean)")]:
        fig = plt.figure(figsize=(8, 5.5))
        total_best = np.zeros_like(nu_plot)
        total_input = np.zeros_like(nu_plot)
        total_draws = np.zeros((draws.shape[0], nu_plot.size), dtype=float)

        for ci, (spec, comp) in enumerate(model._comps):
            stokes_out = tuple(getattr(comp, "stokes_out", ("I",)))
            emits_polarization = ("Q" in stokes_out) or ("U" in stokes_out)
            if mode == "P" and (not emits_polarization or spec.name == "spinningdust"):
                continue

            template = model.templates[spec.template_name]
            comp_best = np.zeros_like(nu_plot)
            comp_input = np.zeros_like(nu_plot)
            comp_draw = np.zeros((draws.shape[0], nu_plot.size), dtype=float)

            for k, nu in enumerate(nu_plot):
                i_best, p_best = _eval_component_region_mean(spec, comp, template, float(nu), params0)
                val_best = i_best if mode == "I" else p_best
                comp_best[k] = val_best
                comp_input[k] = val_best
                for j, pd in enumerate(draw_params):
                    try:
                        i_d, p_d = _eval_component_region_mean(spec, comp, template, float(nu), pd)
                    except ValueError:
                        continue
                    comp_draw[j, k] = i_d if mode == "I" else p_d

            comp_draw_abs = np.abs(comp_draw)
            lo = np.percentile(comp_draw_abs, 0.15, axis=0)
            hi = np.percentile(comp_draw_abs, 99.85, axis=0)
            color = comp_colors[spec.name]
            plt.fill_between(nu_plot, np.clip(lo, 1e-30, np.inf), np.clip(hi, 1e-30, np.inf), color=color, alpha=0.18)
            plt.plot(nu_plot, np.clip(np.abs(comp_best), 1e-30, np.inf), color=color, lw=2, label=f"{spec.name} best-fit")
            plt.plot(nu_plot, np.clip(np.abs(comp_input), 1e-30, np.inf), color=color, lw=1.4, ls="--", label=f"{spec.name} input")

            total_best += comp_best
            total_input += comp_input
            total_draws += comp_draw

        total_draws_abs = np.abs(total_draws)
        total_lo = np.percentile(total_draws_abs, 0.15, axis=0)
        total_hi = np.percentile(total_draws_abs, 99.85, axis=0)
        plt.fill_between(nu_plot, np.clip(total_lo, 1e-30, np.inf), np.clip(total_hi, 1e-30, np.inf), color="k", alpha=0.12)
        plt.plot(nu_plot, np.clip(np.abs(total_best), 1e-30, np.inf), color="k", lw=2.5, label="sum best-fit")
        plt.plot(nu_plot, np.clip(np.abs(total_input), 1e-30, np.inf), color="k", lw=1.5, ls="--", label="sum input")

        for nu, mk in marker_by_freq.items():
            ytot = np.interp(float(nu), nu_plot, np.clip(np.abs(total_best), 1e-30, np.inf))
            plt.scatter([nu], [ytot], marker=mk, color="k", s=40, zorder=5)

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Frequency [GHz]")
        plt.ylabel(ylabel)
        plt.title(f"{fitdata.map_names[0]}... {mode}-mode SED (region mean)")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        suffix = "intensity" if mode == "I" else "polarized"
        plt.savefig(reg_out / f"sed_{suffix}.png", dpi=180)
        plt.close(fig)


def run_fisher(
    fitter_yaml: Path,
    data_yaml: Path,
    templates_yaml: Path,
    regions_h5: Path,
    processed_h5: Path,
    out_dir: Path,
    run_name: str,
    region_ids: Optional[List[str]],
    overwrite: bool = False,
    dry_run: bool = False,
    deriv_method: str = "finite",
    stepsize: float = 1e-3,
    num_points: int = 5,
    extrapolation: Optional[str] = "ridders",
    levels: int = 4,
    n_workers: int = 1,
) -> None:
    del deriv_method, stepsize, num_points, extrapolation, levels, n_workers, overwrite

    fitter_info = load_yaml(fitter_yaml)["fitter"]
    data_info = load_yaml(data_yaml)

    dataset_sets = _collect_dataset_sets(fitter_info)
    set_names = list(dataset_sets.keys())

    region_tag = str(fitter_info.get("regions_tag", "v001"))
    region_group = str(fitter_info.get("regions_group", "gal_plus_high_1"))

    regions = RegionsIO(regions_h5, region_tag).load_regions(region_group)
    templates = load_templates_config(templates_yaml, processed_h5)

    if not region_ids:
        region_ids = regions.region_names

    components, param0, _widths0, global_prior, _gain_params = build_components_from_yaml(fitter_info)
    fit_stokes = _resolve_fit_stokes(fitter_info)
    param_names = list(param0.keys())
    md_params = _magnetic_dust_params(param_names)
    include_gain_params_in_corner = bool(fitter_info.get("include_gain_params_in_corner", False))

    out_root = Path(fitter_info.get("out_dir", out_dir)) / str(fitter_info.get("sims_tag", "v001")) / run_name
    _ensure_dir(out_root)

    if dry_run:
        print("[DRY RUN] Fisher set names:", set_names)
        print("[DRY RUN] Regions:", region_ids)
        print("[DRY RUN] Params:", param_names)
        print("[DRY RUN] Out:", out_root)
        return

    comparisons: Dict[str, Dict[str, Any]] = {}
    inversion_cfg = _fisher_inversion_config(fitter_info)

    for set_name, targets in dataset_sets.items():
        local_fitter = dict(fitter_info)
        local_fitter["targets"] = list(targets)

        resolved_targets = _resolve_target_entries(local_fitter, data_info, processed_h5)

        mapio_cache: Dict[Tuple[str, str], MapIO] = {}
        maps = {}
        log.info("[%s] Resolved target inputs (source/group/freq/calerr):", set_name)
        for map_name in local_fitter["targets"]:
            info = resolved_targets[map_name]
            src = info["source"]
            key = (str(src.parent), src.name)
            if key not in mapio_cache:
                mapio_cache[key] = MapIO(data_path=str(src.parent), filename=src.name)
            m = mapio_cache[key].read_map(info["group"])
            m.map_id = map_name
            if info["calerr"] is not None:
                m.calerr = info["calerr"]
            if info["freq_ghz"] is not None:
                m.freq_ghz = float(info["freq_ghz"])
            maps[map_name] = m

        for region_name in region_ids:
            pixels = regions.get_pixels(region_name)
            fitdata = FitData.create_from_dict(
                {map_name: v.slice_map(v, pixels) for map_name, v in maps.items()},
                fit_stokes,
                pixels,
            )

            for j, map_name in enumerate(fitdata.map_names):
                resolved_calerr = resolved_targets[map_name]["calerr"]
                if resolved_calerr is not None:
                    fitdata.calerror[j] = resolved_calerr

            region_templates = {tname: v.slice_template(v, pixels) for tname, v in templates.items()}
            model = Model(components, region_templates, stokes_order=fit_stokes)
            model.set_fitdata(fitdata)
            model.set_paramkeys(list(param0.keys()))
            F = fisher_gain_marginalized(
                model=model,
                fitdata=fitdata,
                params_fid=dict(param0),
                param_names=param_names,
                global_prior=global_prior,
                rel_step=1e-6,
                method="central",
            )
            cov, inversion_diag = _invert_fisher_stably(
                F=F,
                param_names=param_names,
                params0=param0,
                inversion_cfg=inversion_cfg,
            )
            sigma = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
            sigma_map = {p: float(s) for p, s in zip(param_names, sigma)}
            snr_map = _snr_summary(param0, sigma_map, md_params)

            reg_out = out_root / "fisher" / set_name / region_name
            _ensure_dir(reg_out)
            np.save(reg_out / "fisher.npy", F)
            np.save(reg_out / "cov.npy", cov)
            _write_fisher_region_products(
                reg_out=reg_out,
                fisher=F,
                cov=cov,
                param_names=param_names,
                params0=param0,
                sigma_map=sigma_map,
                include_gain_params_in_corner=include_gain_params_in_corner,
            )
            _write_fisher_sed_plots(
                reg_out=reg_out,
                model=model,
                fitdata=fitdata,
                params0=param0,
                cov=cov,
                param_names=param_names,
                global_prior=global_prior,
            )

            summary = {
                "dataset_set": set_name,
                "region": region_name,
                "target_count": len(targets),
                "targets": list(targets),
                "param_names": param_names,
                "fiducial": {p: float(param0[p]) for p in param_names},
                "sigma_1d": sigma_map,
                "covariance": cov.tolist(),
                "magnetic_dust_params": md_params,
                "magnetic_dust_snr": snr_map,
                "fisher_inversion": inversion_diag,
            }
            with open(reg_out / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            comparisons.setdefault(region_name, {})[set_name] = {
                "sigma_1d": sigma_map,
                "covariance": cov.tolist(),
                "magnetic_dust_snr": snr_map,
                "fisher_inversion": inversion_diag,
            }

    side_by_side = {
        "run_name": run_name,
        "dataset_sets": set_names,
        "param_names": param_names,
        "magnetic_dust_params": md_params,
        "regions": comparisons,
    }
    with open(out_root / "fisher" / "dataset_set_comparison.json", "w") as f:
        json.dump(side_by_side, f, indent=2)

    print("\n=== Fisher dataset-set side-by-side summary ===")
    for region_name, region_block in comparisons.items():
        print(f"[{region_name}]")
        for set_name in set_names:
            if set_name not in region_block:
                continue
            snr_map = region_block[set_name]["magnetic_dust_snr"]
            sigma_map = region_block[set_name]["sigma_1d"]
            snr_txt = ", ".join(f"{k}:{v:.3g}" if v is not None else f"{k}:n/a" for k, v in snr_map.items())
            print(f"  - {set_name}: sigma(A_md)={sigma_map.get('A_md', float('nan')):.3g} ; SNR[{snr_txt}]")
