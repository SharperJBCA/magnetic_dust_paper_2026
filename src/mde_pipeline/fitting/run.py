from __future__ import annotations

from pathlib import Path
import csv
from typing import Optional, Dict, Any, List, Tuple
from time import perf_counter
import json
import numpy as np 
import matplotlib.pyplot as plt

from ..utils.logging import get_logger
from ..utils.config import load_yaml
from ..io import raw_maps_readers 
from ..io.maps_io import MapIO, Map  

from ..io.regions_io import RegionsIO
from ..templates.templates import load_templates_config
from ..emission.components import COMPONENTS
from ..qc.qc_plotting import qc_plot_map

from .data_types import Model, ComponentSpec, FitData, build_components_from_yaml
from .priors import NormalPrior, BoundsPrior
from .likelihood import Likelihood
from .emcee_runner import run_emcee_region,ParamVector, load_samples_npz
from .fit_output import write_region_products,write_fisher_region_products

from .fisher import run_fisher,fisher_gain_marginalized

log = get_logger(__name__)


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    sig = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    denom = np.outer(sig, sig)
    corr = np.zeros_like(cov, dtype=float)
    m = denom > 0
    corr[m] = cov[m] / denom[m]
    return corr


def _save_validation_plot(
    out_png: Path,
    param_names: List[str],
    sigma_ratio: np.ndarray,
    pulls: np.ndarray,
    corr_diff: np.ndarray,
) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)

    x = np.arange(len(param_names))
    axs[0].axhline(1.0, color="k", ls="--", lw=1)
    axs[0].plot(x, sigma_ratio, marker="o")
    axs[0].set_title(r"$\sigma_\mathrm{mcmc}/\sigma_\mathrm{fisher}$")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(param_names, rotation=45, ha="right", fontsize=8)

    axs[1].axhline(0.0, color="k", ls="--", lw=1)
    axs[1].plot(x, pulls, marker="o", color="C1")
    axs[1].set_title("Fiducial pull")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(param_names, rotation=45, ha="right", fontsize=8)

    im = axs[2].imshow(corr_diff, cmap="coolwarm", vmin=-1, vmax=1)
    axs[2].set_title("Corr(MCMC) - Corr(Fisher)")
    axs[2].set_xticks(x)
    axs[2].set_yticks(x)
    axs[2].set_xticklabels(param_names, rotation=45, ha="right", fontsize=8)
    axs[2].set_yticklabels(param_names, fontsize=8)
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _resolve_target_entries(
    fitter_info: Dict[str, Any],
    data_info: Dict[str, Any],
    cli_maps_h5: Optional[Path],
) -> Dict[str, Dict[str, Any]]:
    """
    Resolve per-target source/group/calerr using a consistent precedence:
      1) CLI override map path
      2) fitter config defaults/overrides
      3) data.yaml entry
    """
    fitter_default_source = fitter_info.get("sims_h5", fitter_info.get("processed_h5"))
    target_groups = fitter_info.get("target_groups", {})
    calerr_overrides = fitter_info.get("calibration_errors", {})

    gain_prior_sigma: Dict[str, float] = {}
    for g in fitter_info.get("gains", []):
        gparam = str(g.get("param", ""))
        if not gparam.startswith("cal_"):
            continue
        target = gparam.split("cal_", 1)[1]
        priors = g.get("priors", [])
        if priors and priors[0].get("type", "").lower() == "normal":
            params = priors[0].get("params", {})
            if gparam in params and len(params[gparam]) == 2:
                gain_prior_sigma[target] = float(params[gparam][1])

    resolved: Dict[str, Dict[str, Any]] = {}
    missing = [t for t in fitter_info["targets"] if t not in data_info]
    if missing:
        raise KeyError(f"Targets missing from data.yaml: {missing}")

    for target in fitter_info["targets"]:
        dinfo = data_info[target]
        source = cli_maps_h5 or fitter_default_source or dinfo.get("source")
        if source is None:
            raise ValueError(f"No source path could be resolved for target '{target}'")

        group = target_groups.get(target, dinfo.get("group", target))
        calerr = calerr_overrides.get(target, gain_prior_sigma.get(target, dinfo.get("calerr")))
        freq = dinfo.get("freq_ghz")

        resolved[target] = {
            "source": Path(source),
            "group": str(group),
            "calerr": None if calerr is None else float(calerr),
            "freq_ghz": freq,
        }

    return resolved

def regen_region_products_from_npz(
    samples_npz: Path,
    out_run_dir: Path,
    run_name_tag: str,
    maps: dict,            # your loaded maps dict (map_name -> Map)
    templates: dict,       # your loaded templates dict (template_name -> Template)
    components: list,
    Model,
    FitData,
    stokes_order=("I","Q","U"),
    include_lnL_norm=False,
    posterior_predictive_draws=200,
    nside=None,
    make_healpix_maps=False,
):
    result, meta = load_samples_npz(samples_npz)
    region_name = meta["region_name"]
    pixels = meta["pixels"]

    # rebuild fitdata/model using pixels from npz
    fitdata = FitData.create_from_dict(
        {map_name: v.slice_map(v, pixels) for map_name, v in maps.items()},
        list(stokes_order),
        pixels,
    )
    region_templates = {tname: v.slice_template(v, pixels) for tname, v in templates.items()}
    model = Model(components, region_templates, stokes_order=list(stokes_order))

    pv = ParamVector(result.param_names)

    # call the same writer
    return write_region_products(
        out_run_dir=out_run_dir,
        run_name_tag=run_name_tag,
        region_name=region_name,
        fitdata=fitdata,
        model=model,
        result=result,          # EmceeResultLite has the same fields we use
        param_vector=pv,
        include_lnL_norm=include_lnL_norm,
        posterior_predictive_draws=posterior_predictive_draws,
        nside=nside,
        make_healpix_maps=make_healpix_maps,
    )


def run_fit(
    fitter_yaml: Path,
    data_yaml: Path,
    templates_yaml: Path,
    regions_h5: Optional[Path],
    processed_h5: Optional[Path], # can be simulations or real data
    out_dir: Optional[Path],
    run_name: Optional[str],
    region_ids: Optional[List[str]],
    overwrite: bool,
    dry_run: bool,
) -> None:
    overall_t0 = perf_counter()
    fitter_info = load_yaml(fitter_yaml)["fitter"]
    data_info = load_yaml(data_yaml)

    region_tag = str(fitter_info.get("regions_tag", "v001"))
    region_group = str(fitter_info.get("regions_group", "gal_plus_high_1"))

    regions_path = Path(regions_h5) if regions_h5 is not None else Path(fitter_info["regions_h5"])
    maps_h5 = Path(processed_h5) if processed_h5 is not None else Path(
        fitter_info.get("sims_h5", fitter_info["processed_h5"])
    )
    templates_h5 = Path(processed_h5) if processed_h5 is not None else Path(fitter_info["processed_h5"])
    resolved_targets = _resolve_target_entries(fitter_info, data_info, Path(processed_h5) if processed_h5 is not None else None)

    out_root = Path(out_dir) if out_dir is not None else Path(fitter_info["out_dir"])
    run_name = run_name or str(fitter_info.get("run_name", "paper_main"))
    out_run_dir = out_root / run_name

    mode_cfg = fitter_info.get("modes", {})
    run_fisher = bool(mode_cfg.get("run_fisher", True))
    run_mcmc = bool(mode_cfg.get("run_mcmc", True))
    run_postprocess = bool(mode_cfg.get("run_postprocess", False))
    validation_cfg = fitter_info.get("validation", {})
    run_validation = bool(validation_cfg.get("enabled", False))

    if not any([run_fisher, run_mcmc, run_postprocess, run_validation]):
        raise ValueError("No fitting mode enabled. Set at least one of run_fisher/run_mcmc/run_postprocess.")

    log.info(
        "Starting fit run=%s regions_file=%s region_tag=%s suite=%s maps=%s out=%s modes={fisher:%s,mcmc:%s,post:%s}",
        run_name,
        regions_path,
        region_tag,
        region_group,
        maps_h5,
        out_run_dir,
        run_fisher,
        run_mcmc,
        run_postprocess,
    )

    regions = RegionsIO(regions_path, region_tag).load_regions(region_group)
    templates = load_templates_config(templates_yaml, templates_h5) 

    mapio_cache: Dict[Tuple[str, str], MapIO] = {}

    def _load_maps_for_targets(target_cfg: Dict[str, Dict[str, Any]]) -> Dict[str, Map]:
        loaded: Dict[str, Map] = {}
        log.info("Resolved target inputs (source/group/freq/calerr):")
        for map_name in fitter_info["targets"]:
            info = target_cfg[map_name]
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
            loaded[map_name] = m
            log.info(
                "  target=%s source=%s group=%s freq_ghz=%s calerr=%s",
                map_name,
                src,
                info["group"],
                m.freq_ghz,
                m.calerr,
            )
        return loaded

    maps = _load_maps_for_targets(resolved_targets)
    if not region_ids:
        region_ids = regions.region_names


    components, param0, widths0, global_prior, gain_param_names = build_components_from_yaml(fitter_info)
    #global_prior = None
    lnlike_obj = Likelihood(components, None)

    mcmc_cfg = fitter_info.get("mcmc", {})

    def _run_mcmc_region(fitdata: FitData, model: Model, seed: int):
        return run_emcee_region(
            lnpost_obj=lnlike_obj,
            fitdata=fitdata,
            model=model,
            params0=param0,
            widths0=widths0,
            extra_param_names=gain_param_names,
            components=components,
            burn_steps=int(mcmc_cfg.get("burn_steps", 500)),
            burn_stages=int(mcmc_cfg.get("burn_stages", 1)),
            prod_steps=int(mcmc_cfg.get("prod_steps", 2500)),
            prod_burnin=int(mcmc_cfg.get("prod_burnin", 1500)),
            thin=int(mcmc_cfg.get("thin", 1)),
            nwalkers=int(mcmc_cfg.get("nwalkers", 64)),
            prune_every=int(mcmc_cfg.get("prune_every", 250)),
            seed=seed,
        )

    region_templates = {}
    run_rows = []
    for region_name in region_ids:
        region_t0 = perf_counter()
        log.info("[REGION:%s] start", region_name)
        pixels = regions.get_pixels(region_name)

        fitdata = FitData.create_from_dict( {map_name:v.slice_map(v,pixels) for map_name, v in maps.items()},
                                           ["I","Q","U"],
                                           pixels)
        for j, map_name in enumerate(fitdata.map_names):
            resolved_calerr = resolved_targets[map_name]["calerr"]
            if resolved_calerr is not None:
                fitdata.calerror[j] = resolved_calerr

        region_templates[region_name] = {template_name:v.slice_template(v,pixels) for template_name, v in templates.items()}
        model = Model(components, region_templates[region_name], stokes_order=['I','Q','U'])

        # ---- Fisher sanity check (before MCMC) ----
        # Build fiducial params dict including gains 

        # fiducial params = param0 (your science params in sampling space)
        params_fid = dict(param0)
        param_names = list(params_fid.keys())

        if run_fisher:
            fisher_t0 = perf_counter()
            F = fisher_gain_marginalized(
                model=model,
                fitdata=fitdata,
                params_fid=params_fid,
                param_names=param_names,
                global_prior=global_prior,   # optional
                rel_step=1e-6,
                method="central",
            )

            cov = np.linalg.pinv(F)
            sig = np.sqrt(np.clip(np.diag(cov), 0, np.inf))

            log.info("[REGION:%s][MODE:FISHER] writing outputs", region_name)
            fisher_result = {}
            for n, s in zip(param_names, sig):
                log.info("[REGION:%s][MODE:FISHER] sigma(%s)=%.4g", region_name, n, s)
                fisher_result[n] = s

            pv = ParamVector(param_names)
            fisher_summary = write_fisher_region_products(
                out_run_dir=out_run_dir,
                run_name_tag=run_name,
                region_name=region_name,
                fitdata=fitdata,
                model=model,
                fisher_result=fisher_result,
                fisher_cov=cov,
                params0=param0,
                param_vector=pv,
            )
            run_rows.append({
                "run_id": run_name,
                "region": region_name,
                "mode": "fisher",
                "chi2": "",
                "red_chi2": "",
                "AIC": "",
                "BIC": "",
                **{f"sigma_{k}": v for k, v in fisher_result.items()},
                **{f"fid_{k}": v for k, v in params_fid.items() if not k.startswith("cal_")},
            })
            log.info("[REGION:%s][MODE:FISHER] done in %.2fs", region_name, perf_counter() - fisher_t0)

        if run_mcmc:
            mcmc_t0 = perf_counter()
            result = _run_mcmc_region(fitdata, model, seed=0)

            pv = ParamVector(result.param_names)

            mcmc_summary = write_region_products(
                out_run_dir=out_run_dir,
                run_name_tag=run_name,
                region_name=region_name,
                fitdata=fitdata,
                model=model,
                result=result,
                param_vector=pv,
                include_lnL_norm=False,
                posterior_predictive_draws=200,  
                nside=None,                      
                make_healpix_maps=False,       
            )
            run_rows.append({
                "run_id": run_name,
                "region": region_name,
                "mode": "mcmc",
                "chi2": mcmc_summary.get("chi2", ""),
                "red_chi2": mcmc_summary.get("red_chi2", ""),
                "AIC": mcmc_summary.get("AIC", ""),
                "BIC": mcmc_summary.get("BIC", ""),
                **{f"best_{k}": v for k, v in mcmc_summary.get("best_params", {}).items() if not k.startswith("cal_")},
            })
            log.info("[REGION:%s][MODE:MCMC] done in %.2fs", region_name, perf_counter() - mcmc_t0)

        if run_postprocess:
            post_t0 = perf_counter()
            samples = out_run_dir / "regions" / region_name / "samples.npz"

            regen_region_products_from_npz(
                samples_npz=samples,
                out_run_dir=out_run_dir,
                run_name_tag=run_name,
                maps=maps,
                templates=templates,
                components=components,
                Model=Model,
                FitData=FitData,
                posterior_predictive_draws=500,
                make_healpix_maps=True,
                nside=16,
            )
            log.info("[REGION:%s][MODE:POSTPROCESS] done in %.2fs", region_name, perf_counter() - post_t0)

        log.info("[REGION:%s] complete in %.2fs", region_name, perf_counter() - region_t0)

    if run_validation:
        val_regions = validation_cfg.get("region_ids") or region_ids
        grid_ids = [str(g) for g in validation_cfg.get("grid_ids", [])]
        if not grid_ids:
            raise ValueError("fitter.validation.enabled=true requires fitter.validation.grid_ids")

        val_target_groups = validation_cfg.get("target_groups_by_grid", {})
        val_grid_sources = validation_cfg.get("grid_sources", {})
        val_tol = validation_cfg.get("tolerances", {})
        sigma_ratio_min = float(val_tol.get("sigma_ratio_min", 0.5))
        sigma_ratio_max = float(val_tol.get("sigma_ratio_max", 1.5))
        corr_diff_max = float(val_tol.get("corr_diff_norm_max", 1.0))
        pull_max = float(val_tol.get("pull_abs_max", 3.0))
        fail_on_exceed = bool(validation_cfg.get("fail_on_exceed", False))

        val_mcmc_cfg = validation_cfg.get("mcmc", {})
        mcmc_cfg_backup = dict(mcmc_cfg)
        mcmc_cfg.update(val_mcmc_cfg)
        try:
            for grid_id in grid_ids:
                log.info("[VALIDATION][GRID:%s] start", grid_id)
                local_resolved = {k: dict(v) for k, v in resolved_targets.items()}
                if grid_id in val_grid_sources:
                    src = Path(val_grid_sources[grid_id])
                    for mname in local_resolved:
                        local_resolved[mname]["source"] = src
                if grid_id in val_target_groups:
                    for mname, gname in val_target_groups[grid_id].items():
                        if mname in local_resolved:
                            local_resolved[mname]["group"] = str(gname)

                grid_maps = _load_maps_for_targets(local_resolved)
                for region_name in val_regions:
                    pixels = regions.get_pixels(region_name)
                    fitdata = FitData.create_from_dict(
                        {map_name: v.slice_map(v, pixels) for map_name, v in grid_maps.items()},
                        ["I", "Q", "U"],
                        pixels,
                    )
                    for j, map_name in enumerate(fitdata.map_names):
                        resolved_calerr = local_resolved[map_name]["calerr"]
                        if resolved_calerr is not None:
                            fitdata.calerror[j] = resolved_calerr

                    region_templates = {template_name: v.slice_template(v, pixels) for template_name, v in templates.items()}
                    model = Model(components, region_templates, stokes_order=['I', 'Q', 'U'])

                    param_names = list(param0.keys())
                    fisher = fisher_gain_marginalized(
                        model=model,
                        fitdata=fitdata,
                        params_fid=dict(param0),
                        param_names=param_names,
                        global_prior=global_prior,
                        rel_step=1e-6,
                        method="central",
                    )
                    cov_fisher = np.linalg.pinv(fisher)

                    result = _run_mcmc_region(fitdata, model, seed=int(validation_cfg.get("seed", 17)))
                    flat = result.chain.reshape(-1, len(result.param_names))
                    cov_mcmc = np.cov(flat, rowvar=False)
                    means = flat.mean(axis=0)

                    sigma_fisher = np.sqrt(np.clip(np.diag(cov_fisher), 0, np.inf))
                    sigma_mcmc = np.sqrt(np.clip(np.diag(cov_mcmc), 0, np.inf))
                    sigma_ratio = np.divide(sigma_mcmc, sigma_fisher, out=np.full_like(sigma_mcmc, np.nan), where=sigma_fisher > 0)

                    corr_fisher = _cov_to_corr(cov_fisher)
                    corr_mcmc = _cov_to_corr(cov_mcmc)
                    corr_diff = corr_mcmc - corr_fisher
                    corr_diff_norm = float(np.linalg.norm(corr_diff, ord="fro"))

                    param0_vec = np.array([param0[n] for n in result.param_names], dtype=float)
                    pulls = np.divide(means - param0_vec, sigma_mcmc, out=np.full_like(means, np.nan), where=sigma_mcmc > 0)

                    ratio_viol = [result.param_names[i] for i, r in enumerate(sigma_ratio) if not (sigma_ratio_min <= r <= sigma_ratio_max)]
                    pull_viol = [result.param_names[i] for i, p in enumerate(np.abs(pulls)) if p > pull_max]
                    corr_viol = corr_diff_norm > corr_diff_max
                    passed = (not ratio_viol) and (not pull_viol) and (not corr_viol)

                    out_dir_validation = out_run_dir / "validation" / grid_id / str(region_name)
                    out_dir_validation.mkdir(parents=True, exist_ok=True)
                    _save_validation_plot(
                        out_png=out_dir_validation / "comparison.png",
                        param_names=result.param_names,
                        sigma_ratio=sigma_ratio,
                        pulls=pulls,
                        corr_diff=corr_diff,
                    )

                    payload = {
                        "grid_id": grid_id,
                        "region": str(region_name),
                        "param_names": result.param_names,
                        "cov_fisher": cov_fisher.tolist(),
                        "cov_mcmc": cov_mcmc.tolist(),
                        "sigma_ratio_mcmc_over_fisher": {n: float(v) for n, v in zip(result.param_names, sigma_ratio)},
                        "corr_diff_norm_fro": corr_diff_norm,
                        "fiducial_pull": {n: float(v) for n, v in zip(result.param_names, pulls)},
                        "tolerances": {
                            "sigma_ratio_min": sigma_ratio_min,
                            "sigma_ratio_max": sigma_ratio_max,
                            "corr_diff_norm_max": corr_diff_max,
                            "pull_abs_max": pull_max,
                        },
                        "violations": {
                            "sigma_ratio": ratio_viol,
                            "pull_abs": pull_viol,
                            "corr_diff_norm": bool(corr_viol),
                        },
                        "passed": bool(passed),
                    }
                    with open(out_dir_validation / "validation.json", "w") as f:
                        json.dump(payload, f, indent=2)

                    if not passed:
                        msg = (
                            f"[VALIDATION][GRID:{grid_id}][REGION:{region_name}] FAILED "
                            f"violations={{sigma_ratio:{ratio_viol}, pull:{pull_viol}, corr_diff_norm:{corr_viol}}}"
                        )
                        if fail_on_exceed:
                            raise RuntimeError(msg)
                        log.warning(msg)
                    else:
                        log.info("[VALIDATION][GRID:%s][REGION:%s] passed", grid_id, region_name)
        finally:
            mcmc_cfg.clear()
            mcmc_cfg.update(mcmc_cfg_backup)

    log.info("Fit run complete: run=%s regions=%d elapsed=%.2fs", run_name, len(region_ids), perf_counter() - overall_t0)
