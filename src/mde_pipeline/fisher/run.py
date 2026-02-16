from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# DerivKit
from derivkit import DerivativeKit  # robust numerical differentiation

# Your pipeline imports (adjust if paths differ)
from .data_types import FitData, Model
from .param_vector import ParamVector
from .likelihood import Likelihood
from ..io.regions_io import RegionsIO
from ..io.maps_io import MapIO
from ..utils.yaml import load_yaml
from ..templates.loader import load_templates_config
from .build import build_components_from_yaml


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _flatten_pred(pred_2d: np.ndarray) -> np.ndarray:
    """
    pred_2d: shape (nmaps, nstokes*npix)
    -> y: shape (nmaps*nstokes*npix,)
    """
    return np.asarray(pred_2d, dtype=float).reshape(-1)


def _build_noise_from_fitdata(fitdata: FitData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      invN_diag: diagonal of N^{-1} (shape ndata,)
      sigma_diag: diagonal sigma (shape ndata,)
    We assume fitdata.ivar is diagonal inverse-variance per datum.
    """
    invN_diag = np.asarray(fitdata.ivar, dtype=float).reshape(-1)
    # Guard: avoid division by zero
    sigma2 = np.where(invN_diag > 0.0, 1.0 / invN_diag, np.inf)
    sigma = np.sqrt(sigma2)
    return invN_diag, sigma


def fisher_from_jacobian(J: np.ndarray, invN_diag: np.ndarray) -> np.ndarray:
    """
    Fisher with diagonal N:
      F = J^T N^{-1} J
    J shape: (ndata, npar)
    invN_diag shape: (ndata,)
    """
    wJ = J * invN_diag[:, None]
    return J.T @ wJ


def invert_fisher(F: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """
    Robust inverse: use pseudo-inverse if ill-conditioned.
    """
    # Try direct inverse first
    try:
        C = np.linalg.inv(F)
        if np.all(np.isfinite(C)):
            return C
    except np.linalg.LinAlgError:
        pass
    # Fallback: pinv
    return np.linalg.pinv(F, rcond=rcond)


def summarize_fisher(F: np.ndarray, C: np.ndarray, param_names: List[str]) -> Dict[str, Any]:
    sig = np.sqrt(np.clip(np.diag(C), 0.0, np.inf))
    # correlation
    denom = np.outer(sig, sig)
    corr = np.where(denom > 0, C / denom, 0.0)
    return {
        "param_names": list(param_names),
        "F": F.tolist(),
        "Cov": C.tolist(),
        "sigma_1d": {p: float(s) for p, s in zip(param_names, sig)},
        "corr": corr.tolist(),
    }


def compute_jacobian_derivkit(
    func,
    theta0: np.ndarray,
    method: str = "finite",
    stepsize: float = 1e-3,
    num_points: int = 5,
    extrapolation: Optional[str] = "ridders",
    levels: int = 4,
    n_workers: int = 1,
) -> np.ndarray:
    """
    Compute Jacobian J_{a,i} = d f_a / d theta_i using DerivKit.

    We do it parameter-by-parameter with DerivativeKit (robust and explicit),
    because DerivativeKit supports vector-valued outputs component-wise.

    Returns:
      J: shape (ndata, npar)
    """
    theta0 = np.asarray(theta0, dtype=float)
    npar = theta0.size

    # Evaluate once to get ndata
    y0 = np.asarray(func(theta0), dtype=float).reshape(-1)
    ndata = y0.size
    J = np.zeros((ndata, npar), dtype=float)

    # Helper: 1D function varying only parameter i
    def f_i(x: float, i: int) -> np.ndarray:
        th = theta0.copy()
        th[i] = x
        return np.asarray(func(th), dtype=float).reshape(-1)

    for i in range(npar):
        dk = DerivativeKit(function=lambda x, ii=i: f_i(x, ii), x0=float(theta0[i]))

        # First derivative wrt scalar x at x0, output is vector (ndata,)
        deriv = dk.differentiate(
            method=method,
            order=1,
            stepsize=stepsize,
            num_points=num_points,
            extrapolation=extrapolation,
            levels=levels,
            n_workers=n_workers,
        )

        J[:, i] = np.asarray(deriv, dtype=float).reshape(-1)

    return J


def run_fisher(
    fitter_yaml: Path,
    data_yaml: Path,
    templates_yaml: Path,
    regions_h5: Path,
    processed_h5: Path,  # can be simulations or real data
    out_dir: Path,
    run_name: str,
    region_ids: Optional[List[str]],
    overwrite: bool = False,
    dry_run: bool = False,
    # Derivative config (can move into YAML later)
    deriv_method: str = "finite",
    stepsize: float = 1e-3,
    num_points: int = 5,
    extrapolation: Optional[str] = "ridders",
    levels: int = 4,
    n_workers: int = 1,
) -> None:
    """
    Fisher-forecast runner mirroring your MCMC runner structure.
    Produces per-region Fisher matrices and 1D sigma summaries.
    """
    fitter_info = load_yaml(fitter_yaml)["fitter"]

    tag = fitter_info.get("regions_tag", "v001")
    sim_tag = fitter_info.get("sims_tag", "v001")

    # Regions + templates
    regions = RegionsIO(regions_h5, tag).load_regions(fitter_info.get("regions_name", "gal_plus_high_1"))
    templates = load_templates_config(templates_yaml, processed_h5)

    # Load maps (real or sim)
    sims_h5 = Path(fitter_info["sims_h5"])
    mapsio = MapIO(data_path=str(sims_h5.parent), filename=sims_h5.name)

    maps = {}
    for map_name in fitter_info["targets"]:
        maps[map_name] = mapsio.read_map(map_name)

    if not region_ids:
        region_ids = regions.region_names

    # Build components and fiducial parameters (theta0)
    components, param0, widths0, global_prior, gain_param_names = build_components_from_yaml(fitter_info)

    # NOTE: For Fisher we usually *exclude* gain parameters unless you explicitly want them.
    # Your Likelihood marginalizes gains; Fisher on that exact marginal likelihood is more involved.
    # Here we simply forecast on the raw model parameters in param0.
    param_names = list(param0.keys())
    pv = ParamVector(param_names)
    theta0 = pv.dict_to_theta(param0)

    # Output layout
    out_dir = Path(fitter_info.get("out_dir", out_dir)) / sim_tag / run_name
    _ensure_dir(out_dir)

    if dry_run:
        print("[DRY RUN] Would run Fisher for regions:", region_ids)
        print("[DRY RUN] Parameters:", param_names)
        print("[DRY RUN] Out:", out_dir)
        return

    for region_name in region_ids:
        pixels = regions.get_pixels(region_name)

        # Build FitData (slice maps to region)
        fitdata = FitData.create_from_dict(
            {map_name: v.slice_map(v, pixels) for map_name, v in maps.items()},
            ["I", "Q", "U"],
            pixels,
        )

        # Slice templates too
        region_templates = {tname: v.slice_template(v, pixels) for tname, v in templates.items()}
        model = Model(components, region_templates, stokes_order=["I", "Q", "U"])

        # Forward model: y(theta)
        def y_of_theta(theta: np.ndarray) -> np.ndarray:
            params = pv.theta_to_dict(theta)
            pred = model.predict(fitdata, params)  # (nmaps, nstokes*npix)
            return _flatten_pred(pred)

        # Build noise weights from fitdata.ivar
        invN_diag, sigma_diag = _build_noise_from_fitdata(fitdata)

        # Derivatives / Jacobian
        J = compute_jacobian_derivkit(
            func=y_of_theta,
            theta0=theta0,
            method=deriv_method,
            stepsize=stepsize,
            num_points=num_points,
            extrapolation=extrapolation,
            levels=levels,
            n_workers=n_workers,
        )

        # Fisher + covariance
        F = fisher_from_jacobian(J, invN_diag)
        C = invert_fisher(F)

        # Save products
        reg_dir = out_dir / region_name
        _ensure_dir(reg_dir)

        np.save(reg_dir / "fisher.npy", F)
        np.save(reg_dir / "cov.npy", C)
        np.save(reg_dir / "jacobian.npy", J)

        summary = summarize_fisher(F, C, param_names)
        with open(reg_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Handy console output
        sig = summary["sigma_1d"]
        print(f"[{region_name}] 1σ:")
        for k, v in sig.items():
            print(f"  {k:20s} {v:.4g}")