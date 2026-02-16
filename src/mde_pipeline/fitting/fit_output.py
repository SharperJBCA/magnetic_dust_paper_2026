from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import numpy as np

# plotting
import matplotlib.pyplot as plt

try:
    import corner
except Exception:
    corner = None

try:
    import healpy as hp
except Exception:
    hp = None

STOKES_COLORS = {
    "I":"C0",
    "Q":"C1",
    "U":"C2"
}

# ---------- small helpers ----------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def robust_percentiles(x: np.ndarray, q=(16, 50, 84)) -> Tuple[float, float, float]:
    a, b, c = np.percentile(x, q)
    return float(a), float(b), float(c)

def reshape_fitdata(fitdata) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      data: (nfreq, nstokes, npix)
      ivar: (nfreq, nstokes, npix)
    """
    nfreq = fitdata.data.shape[0]
    nstokes = len(fitdata.stokes)
    npix = fitdata.pixels.size
    data = fitdata.data.reshape(nfreq, nstokes, npix)
    ivar = fitdata.ivar.reshape(nfreq, nstokes, npix)
    return data, ivar

def compute_lnL_diag_gauss_from_chi2(chi2: float, include_norm: bool, ivar_flat: np.ndarray) -> float:
    """
    lnL = -0.5 * [chi2 + sum(log(2πσ^2))]
    using ivar = 1/σ^2.
    """
    if not include_norm:
        return float(-0.5 * chi2)
    return float(-0.5 * (chi2 + ivar_flat.size * np.log(2.0 * np.pi) - np.sum(np.log(ivar_flat))))

def aic_bic(lnL_hat: float, k: int, n: int) -> Tuple[float, float]:
    aic = 2.0 * k - 2.0 * lnL_hat
    bic = float(k) * np.log(float(n)) - 2.0 * lnL_hat
    return float(aic), float(bic)

# ---------- fisher writer --------

def write_fisher_region_products(
    out_run_dir: Path,
    run_name_tag: str,
    region_name: str,
    fitdata,
    model,
    fisher_result, 
    fisher_cov,
    params0,
    param_vector  # ParamVector(result.param_names)
) -> Dict[str,Any]: 
    # dirs
    base = ensure_dir(out_run_dir / "regions" / region_name)
    d_fisher = ensure_dir(base / "fisher")

    # Save fisher data
    np.savez_compressed(
        d_fisher / "fisher.npz",
        fisher_result=fisher_result,
        fisher_cov=fisher_cov,
        params0=params0,
        data=fitdata.data,
        ivar=fitdata.ivar,
    )
    # Create a corner plot 
    means = np.array([v for k,v in params0.items()])
    names = list(params0.keys())
    flat_plot = np.random.multivariate_normal(means, fisher_cov, 5000).T

    if corner is not None:
        corner_data = [] 
        corner_labels = [] 
        for fp, lbl in zip(flat_plot, names):
            if not 'cal' in lbl:
                corner_data.append(fp)
                corner_labels.append(lbl)


        corner_data = np.array(corner_data).T
        fig = corner.corner(corner_data, labels=corner_labels, show_titles=True)
        fig.savefig(d_fisher / "corner.png", dpi=200)
        plt.close(fig)
# ---------- main writer ----------

def write_region_products(
    out_run_dir: Path,
    run_name_tag: str,
    region_name: str,
    fitdata,
    model,
    result,
    param_vector,  # ParamVector(result.param_names)
    include_lnL_norm: bool = False,
    corner_max_points: int = 20000,
    posterior_predictive_draws: int = 0,  # e.g. 200 for band in spectrum
    nside: Optional[int] = None,          # required for healpix residual maps
    make_healpix_maps: bool = False,
) -> Dict[str, Any]:
    """
    Writes an inspection + archival bundle for one region.
    Returns a summary dict (also written to JSON).
    """
    # dirs
    base = ensure_dir(out_run_dir / "regions" / region_name)
    d_bestfit = ensure_dir(base / "bestfit")
    d_maps = ensure_dir(base / "resid_maps")
    d_mvd = ensure_dir(base / "model_vs_data")

    # unpack
    chain = result.chain                       # (nsteps, nwalkers, ndim)
    logp = result.log_prob                     # (nsteps, nwalkers)
    acc = result.acceptance_fraction           # (nwalkers,)
    names = result.param_names
    ndim = len(names)

    # flatten samples for summaries/plots
    flat = chain.reshape(-1, ndim)             # (nsteps*nwalkers, ndim)
    flat_logp = logp.reshape(-1)

    # best params
    best_params = param_vector.theta_to_dict(result.best_theta)
    best_pred = model.predict(fitdata, best_params)  # same shape as fitdata.data

    # chi2 + lnL
    resid = fitdata.data - best_pred
    chi2 = float(np.sum(resid * resid * fitdata.ivar))
    n = int(fitdata.data.size)
    k = int(ndim)
    lnL_hat = compute_lnL_diag_gauss_from_chi2(chi2, include_lnL_norm, fitdata.ivar.reshape(-1))
    AIC, BIC = aic_bic(lnL_hat, k, n)
    red_chi2 = chi2 / max(1, (n - k))

    # ---- save samples ----
    np.savez_compressed(
        base / "samples.npz",
        chain=chain,
        log_prob=logp,
        acceptance_fraction=acc,
        best_theta=result.best_theta,
        best_log_prob=result.best_log_prob,
        param_names=np.array(names, dtype="U"),
        region_name=np.array(region_name),
        frequencies_ghz=np.array(fitdata.frequencies_ghz, dtype=float),
        stokes=np.array(fitdata.stokes, dtype="U"),
        pixels=np.array(fitdata.pixels, dtype=np.int64),
    )

    # ---- summary stats ----
    per_param = {}
    for i, pname in enumerate(names):
        p16, p50, p84 = robust_percentiles(flat[:, i], (16, 50, 84))
        per_param[pname] = {
            "p16": p16,
            "p50": p50,
            "p84": p84,
            "minus": p50 - p16,
            "plus": p84 - p50,
            "map": float(result.best_theta[i]),
        }

    summary = {
        "region": region_name,
        "n": n,
        "k": k,
        "chi2": chi2,
        "red_chi2": red_chi2,
        "lnL_hat": lnL_hat,
        "AIC": AIC,
        "BIC": BIC,
        "acceptance_fraction": {
            "median": float(np.median(acc)),
            "min": float(np.min(acc)),
            "max": float(np.max(acc)),
        },
        "param_names": names,
        "params": per_param,
        "best_params": best_params,
        "frequencies_ghz": [float(x) for x in fitdata.frequencies_ghz],
        "stokes": list(fitdata.stokes),
        "npix": int(fitdata.pixels.size),
    }

    with open(base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- save bestfit arrays ----
    np.savez_compressed(
        d_bestfit / "prediction_residuals.npz",
        prediction=best_pred,
        residuals=resid,
        z=resid * np.sqrt(fitdata.ivar),
        data=fitdata.data,
        ivar=fitdata.ivar,
    )
    with open(d_bestfit / "params.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # ---- corner ----
    if corner is not None:
        # subsample for speed/size if needed
        if flat.shape[0] > corner_max_points:
            idx = np.random.default_rng(0).choice(flat.shape[0], size=corner_max_points, replace=False)
            flat_plot = flat[idx]
        else:
            flat_plot = flat

        corner_data = [] 
        corner_labels = [] 
        for fp, lbl in zip(flat_plot.T, names):
            if True:#not 'cal' in lbl:
                corner_data.append(fp)
                corner_labels.append(lbl)

        corner_data = np.array(corner_data).T
        fig = corner.corner(corner_data, labels=corner_labels, show_titles=True)
        fig.savefig(base / "corner.png", dpi=200)
        plt.close(fig)

    # ---- trace plots ----
    ndim_no_gain = len(corner_labels) 
    fig, axes = plt.subplots(ndim_no_gain, 1, figsize=(10, max(2.5, 1.6 * ndim_no_gain)), sharex=True)
    if ndim == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(chain[:, :, i], alpha=0.25, linewidth=0.5)
        ax.set_ylabel(names[i])
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(base / "trace.png", dpi=200)
    plt.close(fig)

    # ---- residual histogram (normalized) ----
    z = (resid * np.sqrt(fitdata.ivar)).reshape(-1)
    z = z[np.isfinite(z)]
    fig = plt.figure(figsize=(6, 4))
    plt.hist(z, bins=80, density=True)
    plt.xlabel(r"z = (d - m) / $\sigma$")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(base / "resid_hist.png", dpi=200)
    plt.close(fig)

    # ---- spectrum plot (region-averaged I/Q/U) ----
    data_3d, ivar_3d = reshape_fitdata(fitdata)
    pred_3d = best_pred.reshape(data_3d.shape)

    freqs = np.asarray(fitdata.frequencies_ghz, dtype=float)
    stokes = list(fitdata.stokes)

    # inverse-variance weighted mean over pixels for each freq/stokes
    def ivw_mean(y, w):
        wsum = np.sum(w, axis=-1)
        ysum = np.sum(y * w, axis=-1)
        mu = np.where(wsum > 0, ysum / wsum, np.nan)
        sig = np.where(wsum > 0, 1.0 / np.sqrt(wsum), np.nan)
        return mu, sig

    data_mu, data_sig = ivw_mean(data_3d, ivar_3d)   # (nfreq, nstokes)
    pred_mu, _ = ivw_mean(pred_3d, ivar_3d)

    fig = plt.figure(figsize=(7, 5))
    freq_sort = np.argsort(freqs)
    freqs = freqs[freq_sort]
    data_mu = data_mu[freq_sort]
    pred_mu = pred_mu[freq_sort]
    data_sig = data_sig[freq_sort]
    for si, s in enumerate(stokes):
        plt.errorbar(freqs, data_mu[:, si], yerr=data_sig[:, si], fmt="o", label=f"data {s}",color=STOKES_COLORS[s], capsize=3)
        plt.plot(freqs, pred_mu[:, si], "-", label=f"model {s}",color=STOKES_COLORS[s])
    plt.xlabel("Frequency [GHz]")
    plt.ylabel("Region-mean Tb")
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(d_mvd / "spectrum.png", dpi=200)
    plt.close(fig)

    # ---- posterior predictive band on spectrum (optional) ----
    if posterior_predictive_draws and posterior_predictive_draws > 0:
        rng = np.random.default_rng(0)
        ndraw = min(posterior_predictive_draws, flat.shape[0])
        draw_idx = rng.choice(flat.shape[0], size=ndraw, replace=False)

        # collect predicted region-means for each draw
        pp = np.zeros((ndraw, freqs.size, len(stokes)), dtype=float)
        for j, idx in enumerate(draw_idx):
            p = param_vector.theta_to_dict(flat[idx])
            pr = model.predict(fitdata, p).reshape(data_3d.shape)
            mu, _ = ivw_mean(pr, ivar_3d)
            pp[j] = mu

        lo = np.percentile(pp, 16, axis=0)
        hi = np.percentile(pp, 84, axis=0)

        freq_sort = np.argsort(freqs)
        freqs = freqs[freq_sort]
        data_mu = data_mu[freq_sort]
        pred_mu = pred_mu[freq_sort]
        data_sig = data_sig[freq_sort]
        lo = lo[freq_sort]
        hi = hi[freq_sort]
        fig = plt.figure(figsize=(7, 5))
        for si, s in enumerate(stokes):
            plt.errorbar(freqs, data_mu[:, si], yerr=data_sig[:, si], fmt="o", label=f"data {s}",
                         color=STOKES_COLORS[s],capsize=3)
            plt.plot(freqs, pred_mu[:, si], "-", label=f"model {s}", lw=3,
                         color=STOKES_COLORS[s],alpha=0.7)
            plt.fill_between(freqs, lo[:, si], hi[:, si], alpha=0.2,
                         color=STOKES_COLORS[s])
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("RMS [K]")
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.tight_layout()
        plt.savefig(d_mvd / "spectrum_pp_band.png", dpi=200)
        plt.close(fig)

    # ---- healpix maps (optional) ----
    if make_healpix_maps:
        if hp is None:
            raise RuntimeError("healpy not available but make_healpix_maps=True")
        if nside is None:
            raise ValueError("Need nside to render healpix maps for a region.")

        # build full-sky arrays (UNSEEN outside region)
        npix_full = hp.nside2npix(nside)
        # data_3d: (nfreq, nstokes, npix_region)
        # pred_3d: same
        resid_3d = data_3d - pred_3d
        z_3d = resid_3d * np.sqrt(ivar_3d)

        for fi, nu in enumerate(freqs):
            for si, s in enumerate(stokes):
                for kind, arr in [
                    ("data", data_3d[fi, si]),
                    ("model", pred_3d[fi, si]),
                    ("resid", resid_3d[fi, si]),
                    ("z", z_3d[fi, si]),
                ]:
                    m = np.full(npix_full, hp.UNSEEN, dtype=float)
                    m[fitdata.pixels] = arr

                    fig = plt.figure(figsize=(8, 4.5))
                    hp.mollview(m, title=f"{region_name} {nu:.2f} GHz {s} {kind}", fig=fig.number)
                    plt.savefig(d_maps / f"{nu:.2f}GHz_{s}_{kind}.png", dpi=200)
                    plt.close(fig)

    return summary
