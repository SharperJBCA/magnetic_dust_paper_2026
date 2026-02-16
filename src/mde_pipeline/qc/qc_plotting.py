from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


def _safe_percentiles(x: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (0.0, 1.0)
    vmin, vmax = np.nanpercentile(x, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        mu = float(np.nanmedian(x)) if x.size else 0.0
        return (mu - 1.0, mu + 1.0)
    return (float(vmin), float(vmax))


def _valid_pixels(m, arr: np.ndarray, qc: Dict[str, Any]) -> np.ndarray:
    """
    Return boolean mask of VALID pixels.
    qc['mask_is_bad']=True means m.mask==True indicates bad pixels.
    """
    valid = np.isfinite(arr) & (arr != hp.UNSEEN)

    mask = getattr(m, "mask", None)
    if isinstance(mask, np.ndarray) and mask.size == arr.size and mask.size > 0:
        mask_is_bad = bool(qc.get("mask_is_bad", True))  # your convention: True=bad
        if mask_is_bad:
            valid &= ~mask.astype(bool)
        else:
            valid &= mask.astype(bool)

    return valid


def _plot_healpix(
    arr: np.ndarray,
    title: str,
    outfile: Path,
    qc: Dict[str, Any],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    proj = str(qc.get("projection", "mollview")).lower()
    unit = qc.get("unit_label", "")

    fig_w = float(qc.get("fig_w", 10.0))
    fig_h = float(qc.get("fig_h", 5.2))
    dpi = int(qc.get("dpi", 140))

    plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    common = dict(
        title=title,
        unit=unit,
        min=vmin,
        max=vmax,
        xsize=int(qc.get("xsize", 1600)),
        cbar=True,
        notext=not bool(qc.get("show_text", True)),
        norm='hist'
    )

    # Choose projection
    if proj == "cartview":
        hp.cartview(arr, **common)
    elif proj == "orthview":
        hp.orthview(arr, **common)
    else:
        hp.mollview(arr, **common)

    if bool(qc.get("graticule", True)):
        hp.graticule()

    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()


def qc_plot_map(m, qc: Dict[str, Any], out_dir: Path) -> None:
    """
    Quick-look QC plots for a processed Healpix Map object.

    Parameters
    ----------
    m : Map-like
        Expected attrs: map_id, I, (Q,U), (II,QQ,UU), mask, unit, freq_ghz, fwhm_arcmin, nside, pol_convention, meta
    qc : dict
        Options, e.g.
          enabled: bool
          projection: "mollview"|"cartview"|"orthview"
          percentiles: [1,99]
          make_hist: bool
          var_display: "log10"|"sqrt"|"linear"
          mask_is_bad: bool  (True means mask==True => bad)
    out_dir : Path
        Directory to write pngs and summary.
    """
    if not bool(qc.get("enabled", True)):
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Defaults
    p_lo, p_hi = qc.get("percentiles", [1, 99])
    p_lo = float(p_lo)
    p_hi = float(p_hi)

    # A little summary text is genuinely helpful when debugging
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w") as f:
        f.write(f"map_id: {getattr(m, 'map_id', '')}\n")
        for k in ["unit", "freq_ghz", "fwhm_arcmin", "nside", "coord", "calerr", "pol_convention"]:
            if hasattr(m, k):
                f.write(f"{k}: {getattr(m, k)}\n")
        meta = getattr(m, "meta", {}) or {}
        if isinstance(meta, dict) and meta:
            f.write("\nmeta:\n")
            for kk in sorted(meta.keys()):
                try:
                    f.write(f"  {kk}: {meta[kk]}\n")
                except Exception:
                    f.write(f"  {kk}: <unprintable>\n")

    # Helper to plot any field
    def plot_field(field: str, arr: np.ndarray, is_variance: bool = False) -> None:
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return

        title_prefix = str(qc.get("title_prefix", "")).strip()
        map_id = getattr(m, "map_id", "")
        title = f"{title_prefix}{map_id}  {field}".strip()

        # Build display array (variance can be log/sqrt)
        disp = arr.astype(np.float64, copy=True)

        valid = _valid_pixels(m, disp, qc)

        if is_variance:
            mode = str(qc.get("var_display", "log10")).lower()
            if mode == "sqrt":
                disp[valid] = np.sqrt(np.maximum(disp[valid], 0.0))
                qc_local_unit = "sqrt(var)"
            elif mode == "linear":
                qc_local_unit = "var"
            else:
                # log10
                disp[valid] = np.log10(np.maximum(disp[valid], 1e-30))
                qc_local_unit = "log10(var)"
            # set invalid to UNSEEN for healpy plotting
            disp[~valid] = hp.UNSEEN
            vmin, vmax = _safe_percentiles(disp[valid], p_lo, p_hi)
            qc_plot = dict(qc)
            qc_plot["unit_label"] = qc_local_unit
        else:
            disp[~valid] = hp.UNSEEN
            vmin, vmax = _safe_percentiles(disp[valid], p_lo, p_hi)
            qc_plot = dict(qc)
            qc_plot["unit_label"] = getattr(m, "unit", "")

        outfile = out_dir / f"{field}.png"
        _plot_healpix(disp, title=title, outfile=outfile, qc=qc_plot, vmin=vmin, vmax=vmax)

        # Optional histogram
        if bool(qc.get("make_hist", True)):
            hist_path = out_dir / f"{field}_hist.png"
            plt.figure(figsize=(7.5, 4.5), dpi=int(qc.get("dpi", 140)))
            vals = disp[valid]
            if vals.size:
                plt.hist(vals, bins=int(qc.get("hist_bins", 200)))
            plt.title(f"{map_id} {field} histogram")
            plt.tight_layout()
            plt.savefig(hist_path, bbox_inches="tight")
            plt.close()

    # Mask plot
    mask = getattr(m, "mask", None)
    if isinstance(mask, np.ndarray) and mask.size == getattr(m, "I", np.empty(0)).size and mask.size > 0:
        # Show "badness" explicitly
        bad = mask.astype(np.float32) if bool(qc.get("mask_is_bad", True)) else (~mask.astype(bool)).astype(np.float32)
        bad_plot = bad.copy()
        bad_plot[~np.isfinite(bad_plot)] = hp.UNSEEN
        _plot_healpix(
            bad_plot,
            title=f"{getattr(m,'map_id','')} mask (bad=1)",
            outfile=out_dir / "mask.png",
            qc={**qc, "unit_label": "bad-mask"},
            vmin=0.0,
            vmax=1.0,
        )

    # Signal maps
    plot_field("I", getattr(m, "I", np.empty(0)), is_variance=False)
    if getattr(m, "Q", np.empty(0)).size > 0:
        plot_field("Q", m.Q, is_variance=False)
    if getattr(m, "U", np.empty(0)).size > 0:
        plot_field("U", m.U, is_variance=False)

    # Variances
    if getattr(m, "II", np.empty(0)).size > 0:
        plot_field("II", m.II, is_variance=True)
    if getattr(m, "QQ", np.empty(0)).size > 0:
        plot_field("QQ", m.QQ, is_variance=True)
    if getattr(m, "UU", np.empty(0)).size > 0:
        plot_field("UU", m.UU, is_variance=True)
