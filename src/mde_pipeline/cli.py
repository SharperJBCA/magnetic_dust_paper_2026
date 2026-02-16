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

from src.mde_pipeline.utils.config import load_yaml
from src.mde_pipeline.utils.paths import _ensure_parent, _default_fits_dir,_default_processed_h5,_default_regions_h5,_default_sims_h5,_default_combine_fits,_default_cmb_fits,_resolve_version_dir

from src.mde_pipeline.utils.logging import setup_logging
from src.mde_pipeline.utils.logging import get_logger

from rich.traceback import install
install(show_locals=False)
app = typer.Typer(
    add_completion=False,
    help="MDE pipeline CLI: preprocessing, simulations, fitting, and paper figures."
)
log = get_logger(__name__)


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
    regions_cfg: Path = typer.Option(Path("configs/regions/gal_plus_high_1.yaml"), "--config", "-c"),
    tag: str = typer.Option("v001", help="Version tag under products/XXX/ (and default processed)."),
    processed: Optional[Path] = typer.Option(None, help="Processed maps HDF5 (defaults by tag)."),
    out: Optional[Path] = typer.Option(None, help="Override output HDF5 path."),
    overwrite: bool = typer.Option(True),
    dry_run: bool = typer.Option(False),
) -> None:
    processed_h5 = processed or _default_processed_h5(tag)
    out_h5 = out or _default_regions_h5(tag)
    log.info("Running regions with config=%s processed_h5=%s out_h5=%s", regions_cfg, processed_h5, out_h5)
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
    processed_path = processed_h5 or _default_processed_h5(tag)

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
        processed_h5=processed_path,
        sims_h5=sims_path,
        out_dir=outdir,
        run_name=run_name,
        region_ids=region_id,
        overwrite=overwrite,
        dry_run=dry_run
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
    raise typer.BadParameter(
        "The 'figures' command is not yet implemented. "
        "Please run preprocessing/regions/simulate/fit commands directly for now."
    )


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
    instruments = Path(configs.get("instruments", "configs/preprocessing/instruments.yaml"))
    preprocess_cfg = Path(configs.get("preprocess", "configs/preprocessing/preprocessing.yaml"))
    regions_cfg = Path(configs.get("regions", "configs/regions/gal_plus_high_1.yaml"))
    templates = Path(configs.get("templates", "configs/templates/templates.yaml"))
    data = Path(configs.get("data", "configs/fitting/data.yaml"))
    sims_cfg = Path(configs.get("sims", "configs/simulations/simulations.yaml"))
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
            run_regions(regions_cfg, processed_h5, regions_h5, overwrite=overwrite, dry_run=dry_run)

        elif name == "simulate":
            run_simulations(
                sims_yaml=sims_cfg,
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
            raise typer.BadParameter(
                "Recipe step 'figures' is not yet implemented. "
                "Remove this step from your recipe for now."
            )

        else:
            raise typer.BadParameter(f"Unknown step name in recipe: {name}")

    typer.echo("== Done ==")


if __name__ == "__main__":
    app()
