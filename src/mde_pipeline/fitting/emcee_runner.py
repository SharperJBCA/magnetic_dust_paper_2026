from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Callable

from scipy.optimize import minimize
import numpy as np

import emcee


# ----------------------------
# Parameter vector utilities
# ----------------------------

@dataclass(frozen=True)
class ParamVector:
    names: List[str]

    def dict_to_theta(self, params: Dict[str, float]) -> np.ndarray:
        return np.array([params[n] for n in self.names], dtype=float)

    def theta_to_dict(self, theta: np.ndarray) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, theta)}


def default_param_order_from_components(components, extra_param_names = []) -> List[str]:
    """
    Builds a stable, deterministic list of *global* parameter names used by components.
    Includes only parameters that appear in spec.params_map values (global names).
    """
    names = []
    seen = set()
    for spec in components:
        # spec.params_map: component_param_name -> global_param_name
        for _, gname in spec.params_map.items():
            if gname not in seen:
                seen.add(gname)
                names.append(gname)


    for gname in extra_param_names:
        if gname not in seen:
            seen.add(gname)
            names.append(gname)
    
    return names


# ----------------------------
# Robust walker pruning
# ----------------------------

def _mad(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis)
    return mad

def prune_and_resample_walkers(
    chain_segment: np.ndarray,      # shape: (nseg, nwalkers, ndim)
    logp_segment: np.ndarray,       # shape: (nseg, nwalkers)
    current_pos: np.ndarray,        # shape: (nwalkers, ndim)
    rng: np.random.Generator,
    z_thresh: float = 6.0,
    lp_thresh: float = 6.0,
    n_tail_draw: int = 50,
    jitter_frac: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify outlier walkers and reset them by sampling from the tail of good walkers.

    Returns:
        new_pos: updated positions (nwalkers, ndim)
        bad_mask: boolean mask of walkers that were reset (nwalkers,)
    """
    nseg, nwalkers, ndim = chain_segment.shape
    if nseg == 0:
        return current_pos, np.zeros(nwalkers, dtype=bool)

    # Use the last n_tail_draw steps from the segment for resampling
    tail = chain_segment[-min(n_tail_draw, nseg):]          # (ntail, nwalkers, ndim)
    tail_lp = logp_segment[-min(n_tail_draw, nseg):]        # (ntail, nwalkers)

    # Walker medians across the segment
    theta_med = np.median(chain_segment, axis=0)            # (nwalkers, ndim)
    lp_med = np.median(logp_segment, axis=0)                # (nwalkers,)

    # Robust center/scale across walkers
    mu = np.median(theta_med, axis=0)                       # (ndim,)
    sig = 1.4826 * _mad(theta_med, axis=0)                  # (ndim,)
    sig = np.where(sig > 0, sig, 1.0)                       # avoid divide-by-zero

    z2 = np.sum(((theta_med - mu) / sig) ** 2, axis=1)      # (nwalkers,)

    lp_mu = np.median(lp_med)
    lp_sig = 1.4826 * _mad(lp_med)
    lp_sig = float(lp_sig) if lp_sig > 0 else 1.0

    # Bad if far in parameter space OR very low probability
    bad = (z2 > (z_thresh ** 2)) | (lp_med < (lp_mu - lp_thresh * lp_sig))

    good_idx = np.where(~bad)[0]
    bad_idx = np.where(bad)[0]
    if good_idx.size == 0 or bad_idx.size == 0:
        return current_pos, bad

    new_pos = np.array(current_pos, copy=True)

    # Resample bad walkers from tail of good walkers
    ntail = tail.shape[0]
    for w in bad_idx:
        gw = int(rng.choice(good_idx))
        t = int(rng.integers(0, ntail))
        base = tail[t, gw, :]

        # Small jitter relative to robust scale
        jitter = rng.normal(scale=jitter_frac, size=ndim) * sig
        new_pos[w, :] = base + jitter

    return new_pos, bad


# ----------------------------
# helpers (optional)
# ----------------------------

def load_samples_npz(npz_path):
    z = np.load(npz_path, allow_pickle=False)
    result = EmceeResult(
        param_names=[str(x) for x in z["param_names"].tolist()],
        chain=z["chain"],
        log_prob=z["log_prob"],
        acceptance_fraction=z["acceptance_fraction"],
        best_theta=z["best_theta"],
        best_log_prob=float(z["best_log_prob"]),
    )
    meta = {
        "region_name": str(z["region_name"]),
        "frequencies_ghz": z["frequencies_ghz"].astype(float),
        "stokes": [str(x) for x in z["stokes"].tolist()],
        "pixels": z["pixels"].astype(np.int64),
    }
    return result, meta


# ----------------------------
# Main runner
# ----------------------------

@dataclass
class EmceeResult:
    param_names: List[str]
    chain: np.ndarray              # (nsteps, nwalkers, ndim) after burn-in removed
    log_prob: np.ndarray           # (nsteps, nwalkers) after burn-in removed
    acceptance_fraction: np.ndarray  # (nwalkers,)
    best_theta: np.ndarray         # (ndim,)
    best_log_prob: float


def run_emcee_region(
    lnpost_obj: Callable[[Dict[str, float], Any, Any], float],
    fitdata: Any,
    model: Any,
    params0: Dict[str, float],
    widths0: Dict[str, float],
    components: List[Any],
    param_names: Optional[List[str]] = None,
    extra_param_names:  Optional[List[str]] = None,
    nwalkers: Optional[int] = None,
    # Staging
    burn_steps: int = 1000,
    burn_stages: int = 3,
    prod_steps: int = 2000,
    prod_burnin: int = 500,
    thin: int = 1,
    # Initialization
    init_scatter: float = 1e-3,
    init_from_prior: Optional[Callable[[np.random.Generator], Dict[str, float]]] = None,
    # Pruning
    prune_every: int = 500,
    prune_z_thresh: float = 6.0,
    prune_lp_thresh: float = 6.0,
    prune_tail_draw: int = 50,
    prune_jitter_frac: float = 1e-4,
    # Misc
    seed: Optional[int] = None,
    progress: bool = True,
    pool=None,
) -> EmceeResult:
    """
    Runs emcee for a single region.

    lnpost_obj should be your existing callable like:
        lnpost_obj(params_dict, fitdata, model) -> float
    where it returns log posterior (priors + lnlike).

    This function wraps it into emcee's expected log_prob(theta).

    Pruning happens during burn-in only, every prune_every steps.
    """
    rng = np.random.default_rng(seed)

    # Decide parameter order
    if param_names is None:
        param_names = default_param_order_from_components(components, extra_param_names)

    param_names = list(params0.keys())
    pv = ParamVector(param_names)

    theta0 = pv.dict_to_theta(params0)
    ndim = theta0.size

    if nwalkers is None:
        nwalkers = max(2 * ndim, 32)  # decent default
    if nwalkers < 2 * ndim:
        raise ValueError(f"nwalkers must be >= 2*ndim for emcee ensemble moves (got {nwalkers}, need {2*ndim}).")

    # Wrap lnpost(params_dict, fitdata, model) -> lnpost(theta)
    def log_prob(theta: np.ndarray) -> float:
        params = pv.theta_to_dict(theta)
        return float(lnpost_obj(params, fitdata, model))

    # Fit for initial parameters 
    def error_func(p):
        params = pv.theta_to_dict(p)
        lp = float(lnpost_obj.chi2(params, fitdata, model))
        if not np.isfinite(lp):
            return 1e100 
        return lp

    #res = minimize(lambda p: error_func(p), theta0, method="L-BFGS-B")
    #theta0 = res.x

    # Build initial walker positions
    p0 = np.zeros((nwalkers, ndim), dtype=float)

    for i,(param_name, param) in enumerate(params0.items()):
        #if param_name in widths0:
        #    p0[:,i] = theta0[i] + widths0[param_name] * rng.normal(size=(nwalkers,))
        #else:
        p0[:,i] = theta0[i] + init_scatter * rng.normal(size=(nwalkers,))


    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)

    # ---------------- burn-in with pruning ----------------
    pos = p0
    state = None

    # Run burn-in in stages so we can prune without too much overhead
    for stage in range(burn_stages):
        nleft = burn_steps
        while nleft > 0:
            nrun = min(prune_every, nleft)
            state = sampler.run_mcmc(pos, nrun, progress=progress)
            pos = state.coords
            nleft -= nrun

            # prune / resample using the most recent segment
            chain_seg = sampler.get_chain()[-nrun:, :, :]
            logp_seg = sampler.get_log_prob()[-nrun:, :]
            pos, bad_mask = prune_and_resample_walkers(
                chain_segment=chain_seg,
                logp_segment=logp_seg,
                current_pos=pos,
                rng=rng,
                z_thresh=prune_z_thresh,
                lp_thresh=prune_lp_thresh,
                n_tail_draw=prune_tail_draw,
                jitter_frac=prune_jitter_frac,
            )

        sampler.reset()

    # ---------------- production ----------------
    state = sampler.run_mcmc(pos, prod_steps, progress=progress)
    chain = sampler.get_chain(thin=thin,discard=prod_burnin)        # (nsteps, nwalkers, ndim)
    logp = sampler.get_log_prob(thin=thin,discard=prod_burnin)      # (nsteps, nwalkers)
    acc = sampler.acceptance_fraction           # (nwalkers,)

    # Best sample by posterior
    best_idx = np.unravel_index(np.argmax(logp), logp.shape)
    best_theta = chain[best_idx[0], best_idx[1], :]
    best_theta = best_theta
    best_lp = float(logp[best_idx[0], best_idx[1]])

    return EmceeResult(
        param_names=param_names,
        chain=chain,
        log_prob=logp,
        acceptance_fraction=acc,
        best_theta=best_theta,
        best_log_prob=best_lp,
    )
