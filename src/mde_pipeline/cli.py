from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import typer
import yaml

from src.mde_pipeline.preprocessing.run import run_preprocess
from src.mde_pipeline.simulations.run import run_simulations
from src.mde_pipeline.combine_cbass_spass.run import run_combine_cbass_spass
from src.mde_pipeline.cmb.run import run_smooth_cmb
from src.mde_pipeline.regions.run import run_regions 
from src.mde_pipeline.fitting.run import run_fit
from src.mde_pipeline.fisher.run import run_fisher
from src.mde_pipeline.fisher.grid_workflow import run_fisher_grid_from_yaml, make_fisher_overlay_corner_plot

from src.mde_pipeline.utils.config import load_yaml
from src.mde_pipeline.utils.paths import _ensure_parent, _default_fits_dir,_default_processed_h5,_default_regions_h5,_default_sims_h5,_default_combine_fits,_default_cmb_fits,_resolve_version_dir

from src.mde_pipeline.utils.logging import setup_logging

from rich.traceback import install
install(show_locals=False)
app = typer.Typer(
    add_completion=False,
    help="MDE pipeline CLI: preprocessing, simulations, fitting, and paper figures."
)


def _print_plan(dry_run: bool, msg: str) -> None:
    prefix = "[DRY-RUN] " if dry_run else ""
    typer.echo(prefix + msg)

# -------------------------
# Stubs for pipeline entrypoints
# Replace these imports with your real modules as you build them.
# -------------------------


def run_figures(
    figures_yaml: Path,
    fit_dir: Path,
    out_dir: Path,
    overwrite: bool,
    dry_run: bool,
) -> None:
    _print_plan(dry_run, f"Figures using {figures_yaml}")
    _print_plan(dry_run, f"  fit_dir={fit_dir}")
    _print_plan(dry_run, f"-> out {out_dir} (overwrite={overwrite})")
    if dry_run:
        return
    _ensure_parent(out_dir / "dummy.txt", dry_run=False)
    # TODO: mde_pipeline.plotting.run_figures(...)
    raise NotImplementedError("Hook up figures runner here.")


# -------------------------
# CLI commands
# -------------------------

@app.callback()
def main(
    log_level: str = typer.Option("INFO", help="Logging level: DEBUG/INFO/WARN/ERROR"),
) -> None:
    setup_logging(log_level)


@app.command()
def preprocess(
    instruments: Path = typer.Option(Path("configs/preprocessing/instruments.yaml"), "--instruments", "-i"),
    preprocess_cfg: Path = typer.Option(Path("configs/preprocessing/preprocessing.yaml"), "--config", "-c"),
    tag: str = typer.Option("v001", help="Version tag under products/processed_maps/"),
    out: Optional[Path] = typer.Option(None, help="Override output HDF5 path."),
    overwrite: bool = typer.Option(False, help="Overwrite existing output group/file."),
    dry_run: bool = typer.Option(False, help="Print actions without writing."),
    only: Optional[List[str]] = typer.Option(
        None,
        "--only",
        help="Restrict to these map_ids (repeatable).")
) -> None:
    out_h5 = out or _default_processed_h5(tag)
    run_preprocess(instruments, 
                   preprocess_cfg, 
                   out_h5, 
                   overwrite=overwrite, 
                   dry_run=dry_run, 
                   only=only)


@app.command()
def regions(
    regions_cfg: Path = typer.Option(Path("configs/regions/regions.yaml"), "--config", "-c"),
    tag: str = typer.Option("v001", help="Version tag under products/XXX/ (and default processed)."),
    processed: Optional[Path] = typer.Option(None, help="Processed maps HDF5 (defaults by tag)."),
    out: Optional[Path] = typer.Option(None, help="Override output HDF5 path."),
    overwrite: bool = typer.Option(True),
    dry_run: bool = typer.Option(False),
) -> None:
    processed_h5 = processed or _default_processed_h5(tag)
    out_h5 = out or _default_regions_h5(tag)
    print(regions_cfg,processed_h5, out_h5)
    run_regions(regions_cfg,processed_h5, out_h5, overwrite=overwrite, dry_run=dry_run)


@app.command()
def simulate(
    sims_cfg: Path = typer.Option(Path("configs/simulations/simulations.yaml"), "--config", "-c"),
    overwrite: bool = typer.Option(False),
    dry_run: bool = typer.Option(False),
) -> None:
    run_simulations(
        sims_cfg,
        overwrite=overwrite,
        dry_run=dry_run
    )


@app.command()
def combine_cbass_spass(
    combine_yaml: Path = typer.Option(Path("configs/combine_cbass_spass/combine.yaml"), "--config", "-c"),
    instruments_yaml: Path = typer.Option(Path("configs/preprocessing/instruments.yaml"), "--instruments", "-i"),
    tag: str = typer.Option("v001", help="Version tag under products/simulations/ (and default processed)."),
    out: Optional[Path] = typer.Option(None, help="Override output HDF5 path."),
    overwrite: bool = typer.Option(True),
    dry_run: bool = typer.Option(False),
) -> None:
    out_fits = out or _default_combine_fits(tag)
    run_combine_cbass_spass(
        combine_yaml,
        instruments_yaml,
        out_fits,
        overwrite,
        dry_run
    )

@app.command() 
def smooth_cmb(
    target_fwhm_arcmin: float = typer.Option(60.0, help="Smoothing target resolution for CMB map."),
    instruments_yaml: Path = typer.Option(Path("configs/preprocessing/instruments.yaml"), "--instruments", "-i"),
    tag: str = typer.Option("v001", help="Version tag under products/cmb_maps/ (and default processed)."),
    out: Optional[Path] = typer.Option(None, help="Override output HDF5 path."),
    overwrite: bool = typer.Option(True),
    dry_run: bool = typer.Option(False),
) -> None:
    out_fits = out or _default_cmb_fits(tag, target_fwhm_arcmin)
    run_smooth_cmb(
        instruments_yaml,
        target_fwhm_arcmin,
        out_fits,
        overwrite,
        dry_run
    )


@app.command()
def fit(
    fitter_cfg: Path = typer.Option(Path("configs/fitting/fitter.yaml"), "--config", "-c"),
    data: Path = typer.Option(Path("configs/fitting/data.yaml"), "--data", "-d"),
    templates: Path = typer.Option(Path("configs/templates/templates.yaml"), "--templates", "-t"),
    tag: str = typer.Option("v001", help="Version tag under products/fits/ (and default processed/regions/sims)."),
    run_name: str = typer.Option("paper_main", help="Name of this fit run (subdir under fits/tag)."),
    regions_h5: Optional[Path] = typer.Option(None, help="Regions HDF5 (defaults by tag)."),
    processed_h5: Optional[Path] = typer.Option(None, help="Processed maps HDF5 (defaults by tag)."),
    sims_h5: Optional[Path] = typer.Option(None, help="Simulations HDF5 (optional; defaults by tag if present)."),
    out_dir: Optional[Path] = typer.Option(None, help="Override output directory."),
    region_id: Optional[List[str]] = typer.Option(None, "--region", help="Restrict to specific region_id(s). Repeatable."),
    overwrite: bool = typer.Option(False),
    dry_run: bool = typer.Option(False),
) -> None:
    outdir = out_dir or _default_fits_dir(tag)
    regions_path = regions_h5 or _default_regions_h5(tag)

    # If user didn't specify sims_h5, you can auto-detect default location.
    sims_path = sims_h5
    default_sim = _default_sims_h5(tag)
    if sims_path is None and default_sim.exists():
        sims_path = default_sim
    run_fit(
        fitter_yaml=fitter_cfg,
        data_yaml=data,
        templates_yaml=templates,
        regions_h5=regions_path,
        processed_h5=sims_path,
        out_dir=outdir,
        run_name=run_name,
        region_ids=region_id,
        overwrite=overwrite,
        dry_run=dry_run
    )



@app.command()
def fisher(
    fitter_cfg: Path = typer.Option(Path("configs/fitting/fitter.yaml"), "--config", "-c"),
    data: Path = typer.Option(Path("configs/fitting/data.yaml"), "--data", "-d"),
    templates: Path = typer.Option(Path("configs/templates/templates.yaml"), "--templates", "-t"),
    tag: str = typer.Option("v001"),
    run_name: str = typer.Option("paper_fisher"),
    regions_h5: Optional[Path] = typer.Option(None),
    processed_h5: Optional[Path] = typer.Option(None),
    out_dir: Optional[Path] = typer.Option(None),
    region_id: Optional[List[str]] = typer.Option(None, "--region"),
    overwrite: bool = typer.Option(False),
    dry_run: bool = typer.Option(False),
    # Derivative knobs
    deriv_method: str = typer.Option("finite"),
    stepsize: float = typer.Option(1e-3),
    num_points: int = typer.Option(5),
    extrapolation: str = typer.Option("ridders"),
    levels: int = typer.Option(4),
    n_workers: int = typer.Option(1),
) -> None:
    outdir = out_dir or _default_fits_dir(tag)
    regions_path = regions_h5 or _default_regions_h5(tag)
    processed_path = processed_h5 or _default_processed_h5(tag)

    run_fisher(
        fitter_yaml=fitter_cfg,
        data_yaml=data,
        templates_yaml=templates,
        regions_h5=regions_path,
        processed_h5=processed_path,
        out_dir=outdir,
        run_name=run_name,
        region_ids=region_id,
        overwrite=overwrite,
        dry_run=dry_run,
        deriv_method=deriv_method,
        stepsize=stepsize,
        num_points=num_points,
        extrapolation=extrapolation,
        levels=levels,
        n_workers=n_workers,
    )



@app.command("fisher-grid")
def fisher_grid(
    config: Path = typer.Option(Path("configs/fisher/grid_workflow.yaml"), "--config", "-c", help="Grid workflow YAML."),
    tag: str = typer.Option("v001"),
    regions_h5: Optional[Path] = typer.Option(None),
    processed_h5: Optional[Path] = typer.Option(None),
    out_dir: Optional[Path] = typer.Option(None),
    reuse_simulation_h5: Optional[Path] = typer.Option(None, "--reuse-simulation-h5", help="Path to a shared simulations.h5 to reuse for all jobs."),
    skip_simulations: bool = typer.Option(False, "--skip-simulations", help="Skip per-job simulation generation and only run Fisher."),
    overwrite: bool = typer.Option(False),
    dry_run: bool = typer.Option(False),
) -> None:
    run_fisher_grid_from_yaml(
        grid_yaml=config,
        tag=tag,
        out_dir=out_dir,
        regions_h5=regions_h5,
        processed_h5=processed_h5,
        reuse_simulation_h5=reuse_simulation_h5,
        skip_simulations=skip_simulations,
        overwrite=overwrite,
        dry_run=dry_run,
    )



@app.command("fisher-grid-config")
def fisher_grid_config(
    config: Path = typer.Option(Path("configs/fisher/grid_workflow.yaml"), "--config", "-c", help="Grid workflow YAML."),
    tag: str = typer.Option("v001"),
    out_dir: Optional[Path] = typer.Option(None),
    regions_h5: Optional[Path] = typer.Option(None),
    processed_h5: Optional[Path] = typer.Option(None),
    reuse_simulation_h5: Optional[Path] = typer.Option(None, "--reuse-simulation-h5", help="Path to a shared simulations.h5 to reuse for all jobs."),
    skip_simulations: bool = typer.Option(False, "--skip-simulations", help="Skip per-job simulation generation and only run Fisher."),
    overwrite: bool = typer.Option(False),
    dry_run: bool = typer.Option(False),
) -> None:
    typer.echo("[DEPRECATED] 'fisher-grid-config' is deprecated; use 'fisher-grid --config ...'.")
    run_fisher_grid_from_yaml(
        grid_yaml=config,
        tag=tag,
        out_dir=out_dir,
        regions_h5=regions_h5,
        processed_h5=processed_h5,
        reuse_simulation_h5=reuse_simulation_h5,
        skip_simulations=skip_simulations,
        overwrite=overwrite,
        dry_run=dry_run,
    )


@app.command("fisher-overlay-corner")
def fisher_overlay_corner(
    runs_root: Path = typer.Option(..., help="Root directory containing the per-run Fisher products."),
    run_a: str = typer.Option(..., help="First run_name (e.g. j0000_modeIQU)."),
    dataset_a: str = typer.Option(..., help="First dataset set (e.g. baseline)."),
    run_b: str = typer.Option(..., help="Second run_name (e.g. j0000_modeQU)."),
    dataset_b: str = typer.Option(..., help="Second dataset set (e.g. baseline_plus_litebird)."),
    region: str = typer.Option(..., help="Region id to load (e.g. highlat1)."),
    out_png: Path = typer.Option(..., help="Output PNG path for the overlay corner."),
    param: Optional[List[str]] = typer.Option(None, "--param", help="Subset of parameters to plot (repeatable)."),
    label: Optional[List[str]] = typer.Option(None, "--label", help="Pretty labels matching --param order (repeatable)."),
    posterior_label: Optional[List[str]] = typer.Option(None, "--posterior-label", help="Legend labels for A and B (repeatable)."),
    transform: Optional[List[str]] = typer.Option(
        None,
        "--transform",
        help="Per-parameter transform mapping formatted as 'param=identity|exp|sigmoid' (repeatable).",
    ),
    true_value: Optional[List[str]] = typer.Option(
        None,
        "--true-value",
        help="Marker value mapping formatted as 'param=value' in transformed/physical space (repeatable).",
    ),
    sample_count: int = typer.Option(5000, help="Number of Gaussian draws per posterior."),
) -> None:
    transform_map: Dict[str, str] = {}
    for item in transform or []:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid --transform '{item}', expected param=transform")
        k, v = item.split("=", 1)
        transform_map[k.strip()] = v.strip()

    true_values_map: Dict[str, float] = {}
    for item in true_value or []:
        if "=" not in item:
            raise typer.BadParameter(f"Invalid --true-value '{item}', expected param=value")
        k, v = item.split("=", 1)
        true_values_map[k.strip()] = float(v.strip())

    make_fisher_overlay_corner_plot(
        runs_root=runs_root,
        run_name_a=run_a,
        dataset_set_a=dataset_a,
        run_name_b=run_b,
        dataset_set_b=dataset_b,
        region=region,
        output_png=out_png,
        params=param,
        pretty_labels=label,
        posterior_labels=posterior_label,
        param_transforms=transform_map or None,
        true_values=true_values_map or None,
        sample_count=sample_count,
    )

@app.command()
def figures(
    figures_cfg: Path = typer.Option(Path("configs/figures.yaml"), "--config", "-c"),
    tag: str = typer.Option("v001", help="Version tag under products/fits/"),
    run_name: str = typer.Option("paper_main", help="Fit run name under products/fits/tag/"),
    out_dir: Path = typer.Option(Path("paper/figures"), "--out", "-o"),
    overwrite: bool = typer.Option(False),
    dry_run: bool = typer.Option(False),
) -> None:
    fit_dir = _default_fits_dir(tag) / run_name
    run_figures(figures_cfg, fit_dir=fit_dir, out_dir=out_dir, overwrite=overwrite, dry_run=dry_run)


@app.command()
def run(
    recipe: Path = typer.Option(..., "--config", "-c", help="Run recipe YAML (orchestrates multiple stages)."),
    dry_run: bool = typer.Option(False),
    overwrite: bool = typer.Option(False),
) -> None:
    try:
        cfg = load_yaml(recipe)
    except FileNotFoundError as e:
        raise typer.BadParameter(f"YAML not found: {e.filename}")

    run_id = cfg.get("run_id", "run")
    typer.echo(f"== Running recipe: {run_id} ==")

    configs = cfg.get("configs", {})
    paths = cfg.get("paths", {})
    steps = cfg.get("steps", [])

    # Resolve common paths
    processed_h5 = Path(paths.get("processed_maps", "products/processed_maps/v001/processed_maps.h5"))
    regions_h5 = Path(paths.get("regions", "products/regions/v001/regions.h5"))
    sims_h5 = Path(paths.get("simulations", "products/simulations/v001/simulations.h5"))
    fits_out = Path(paths.get("fits_out", "products/fits/v001"))

    # Config paths
    instruments = Path(configs.get("instruments", "configs/instruments.yaml"))
    preprocess_cfg = Path(configs.get("preprocess", "configs/preprocess.yaml"))
    regions_cfg = Path(configs.get("regions", "configs/regions.yaml"))
    templates = Path(configs.get("templates", "configs/templates/templates.yaml"))
    data = Path(configs.get("data", "configs/fitting/data.yaml"))
    sims_cfg = Path(configs.get("sims", "configs/simulations.yaml"))
    fitter_cfg = Path(configs.get("fitter", "configs/fitting/fitter.yaml"))
    figures_cfg = Path(configs.get("figures", "configs/figures.yaml"))

    for step in steps:
        name = step.get("name")
        enabled = step.get("enabled", True)
        args = step.get("args", {})

        if not enabled:
            typer.echo(f"- skip {name}")
            continue

        typer.echo(f"- step {name}")

        if name == "preprocess":
            run_preprocess(instruments, preprocess_cfg, processed_h5, overwrite=overwrite, dry_run=dry_run)

        elif name == "regions":
            run_regions(regions_cfg, regions_h5, overwrite=overwrite, dry_run=dry_run)

        elif name == "simulate":
            suite = args.get("suite")
            run_simulations(
                sims_yaml=sims_cfg,
                data_yaml=data,
                templates_yaml=templates,
                processed_h5=processed_h5,
                out_h5=sims_h5,
                suite=suite,
                overwrite=overwrite,
                dry_run=dry_run,
            )

        elif name == "fit":
            run_name = args.get("run_name", "paper_main")
            region_ids = args.get("region_ids")
            run_fit(
                fitter_yaml=fitter_cfg,
                data_yaml=data,
                templates_yaml=templates,
                regions_h5=regions_h5,
                processed_h5=processed_h5,
                sims_h5=sims_h5 if sims_h5.exists() else None,
                out_dir=fits_out,
                run_name=run_name,
                region_ids=region_ids,
                overwrite=overwrite,
                dry_run=dry_run,
            )

        elif name == "figures":
            run_name = args.get("fit_run", "paper_main")
            run_figures(
                figures_yaml=figures_cfg,
                fit_dir=fits_out / run_name,
                out_dir=Path(args.get("out_dir", "paper/figures")),
                overwrite=overwrite,
                dry_run=dry_run,
            )

        else:
            raise typer.BadParameter(f"Unknown step name in recipe: {name}")

    typer.echo("== Done ==")


if __name__ == "__main__":
    app()
