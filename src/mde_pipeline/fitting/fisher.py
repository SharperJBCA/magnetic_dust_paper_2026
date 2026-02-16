from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


@dataclass
class FisherResult:
    param_names: List[str]
    F_data: np.ndarray          # from likelihood only
    F_prior: np.ndarray         # from Gaussian priors only
    F_total: np.ndarray         # sum
    cov: np.ndarray             # inverse(F_total) if invertible
    sigma: np.ndarray           # sqrt(diag(cov))
    corr: np.ndarray            # cov / (sigma_i sigma_j)
    eigvals: np.ndarray         # eigenvalues of F_total


def _as_flat_vector(pred: Any) -> np.ndarray:
    """
    Convert your model.predict(...) output to a flat 1D vector in the same ordering
    as fitdata.data/ivar.

    In your pipeline, prediction is typically a numpy array already (e.g. shape
    (n_map, n_stokes, n_pix) or (n_data,)).
    """
    arr = np.asarray(pred, dtype=float)
    return arr.reshape(-1)


def _get_ivar_vector(fitdata: Any) -> np.ndarray:
    """
    Grab inverse-variance weights and flatten them to match _as_flat_vector ordering.
    """
    if not hasattr(fitdata, "ivar"):
        raise AttributeError("fitdata has no attribute 'ivar'. Need diagonal noise weights for Fisher.")
    ivar = np.asarray(fitdata.ivar, dtype=float).reshape(-1)
    if np.any(~np.isfinite(ivar)):
        raise ValueError("Non-finite values found in fitdata.ivar.")
    if np.any(ivar < 0):
        raise ValueError("Negative values found in fitdata.ivar.")
    return ivar


def _robust_step(p0: float, rel: float = 1e-6, abs_floor: float = 0.0) -> float:
    """
    Relative step in *sampling space* (i.e. your actual parameterization: logs/logits/etc).
    """
    return max(abs_floor, rel * max(1.0, abs(p0)))


def jacobian_finite_difference(
    model: Any,
    fitdata: Any,
    params_fid: Dict[str, float],
    param_names: List[str],
    rel_step: float = 1e-6,
    abs_step_floor: float = 0.0,
    method: str = "central",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute J (n_data, n_params) and mu (n_data,).
    """
    mu = _as_flat_vector(model.predict(fitdata, params_fid))
    n_data = mu.size
    n_par = len(param_names)
    J = np.empty((n_data, n_par), dtype=float)


    for j, name in enumerate(param_names):
        if name not in params_fid:
            raise KeyError(f"Param '{name}' missing from fiducial params dict.")
        p0 = float(params_fid[name])
        h = _robust_step(p0, rel=rel_step, abs_floor=abs_step_floor)

        if method.lower() == "central":
            p_plus = dict(params_fid);  p_plus[name] = p0 + h
            p_minus = dict(params_fid); p_minus[name] = p0 - h
            mu_plus = _as_flat_vector(model.predict(fitdata, p_plus))
            mu_minus = _as_flat_vector(model.predict(fitdata, p_minus))
            J[:, j] = (mu_plus - mu_minus) / (2.0 * h)

        elif method.lower() == "forward":
            p_plus = dict(params_fid); p_plus[name] = p0 + h
            mu_plus = _as_flat_vector(model.predict(fitdata, p_plus))
            J[:, j] = (mu_plus - mu) / h

        else:
            raise ValueError("method must be 'central' or 'forward'.")

    return J, mu


def fisher_from_ivar(J: np.ndarray, ivar: np.ndarray) -> np.ndarray:
    """
    With diagonal noise: F = J^T diag(ivar) J.
    (ivar is inverse-variance weight per datum)
    """
    # weight rows by sqrt(ivar) for numerical stability
    w = np.sqrt(ivar).reshape(-1, 1)
    JW = J * w
    return JW.T @ JW


def fisher_from_normal_priors(global_prior: Any, param_names: List[str]) -> np.ndarray:
    """
    Build a Fisher contribution from Gaussian priors, if you use NormalPrior-like objects.

    We assume NormalPrior stores bounds as {param: (mu, sigma)} like in your earlier snippet,
    OR you can adapt this function to whatever your `global_prior` actually is.

    If you have multiple normal priors, total prior Fisher is sum(1/sigma^2) on the diagonal.
    """
    Fp = np.zeros((len(param_names), len(param_names)), dtype=float)

    if global_prior is None:
        return Fp

    # Try a couple of common patterns without being too magical:
    # 1) global_prior.bounds: dict[param] -> (mu, sigma)
    # 2) global_prior.priors: list of prior objects, some with .bounds
    def add_bounds(bounds: Dict[str, Tuple[float, float]]):
        for k, (mu, sig) in bounds.items():
            if k in param_names and sig is not None and np.isfinite(sig) and sig > 0:
                i = param_names.index(k)
                Fp[i, i] += 1.0 / (sig * sig)

    if hasattr(global_prior, "bounds") and isinstance(global_prior.bounds, dict):
        add_bounds(global_prior.bounds)

    if hasattr(global_prior, "priors"):
        for pr in getattr(global_prior, "priors"):
            if hasattr(pr, "bounds") and isinstance(pr.bounds, dict):
                add_bounds(pr.bounds)

    return Fp


def jacobian_finite_difference_stack(
    model: Any,
    fitdata: Any,
    params_fid: Dict[str, float],
    param_names: List[str],
    rel_step: float = 1e-6,
    abs_step_floor: float = 0.0,
    method: str = "central",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      dmu: (n_par, n_freq, n_dpf)
      mu : (n_freq, n_dpf)
    """
    mu = np.asarray(model.predict(fitdata, params_fid), dtype=float)
    if mu.ndim != 2:
        raise ValueError(f"Expected prediction shape (n_freq, n_dpf), got {mu.shape}")

    n_par = len(param_names)
    dmu = np.empty((n_par,) + mu.shape, dtype=float)

    def step(p0: float) -> float:
        return max(abs_step_floor, rel_step * max(1.0, abs(p0)))

    for i, name in enumerate(param_names):
        if name not in params_fid:
            raise KeyError(f"Param '{name}' missing from fiducial params.")
        p0 = float(params_fid[name])
        h = step(p0)

        if method.lower() == "central":
            p_plus = dict(params_fid);  p_plus[name] = p0 + h
            p_minus = dict(params_fid); p_minus[name] = p0 - h
            mu_plus = np.asarray(model.predict(fitdata, p_plus), dtype=float)
            mu_minus = np.asarray(model.predict(fitdata, p_minus), dtype=float)
            dmu[i] = (mu_plus - mu_minus) / (2.0 * h)

        elif method.lower() == "forward":
            p_plus = dict(params_fid); p_plus[name] = p0 + h
            mu_plus = np.asarray(model.predict(fitdata, p_plus), dtype=float)
            dmu[i] = (mu_plus - mu) / h

        else:
            raise ValueError("method must be 'central' or 'forward'.")

    return dmu, mu


def fisher_gain_marginalized(
    model: Any,
    fitdata: Any,
    params_fid: Dict[str, float],
    param_names: List[str],
    global_prior: Any = None,
    rel_step: float = 1e-6,
    abs_step_floor: float = 0.0,
    method: str = "central",
) -> np.ndarray:
    """
    Fisher for theta after analytically marginalising per-frequency gains g_k with Gaussian priors.

    Uses the identity:
      F_eff = sum_k [ J_k^T W_k J_k  -  (J_k^T W_k m_k)(J_k^T W_k m_k)^T / (m_k^T W_k m_k + 1/sigma_gk^2) ]
    where m_k is ungained model prediction at freq k, J_k is derivative wrt theta, and W_k=diag(ivar_k).
    """

    w = np.asarray(fitdata.ivar, dtype=float)
    if w.ndim != 2:
        raise ValueError(f"Expected fitdata.ivar shape (n_freq, n_dpf), got {w.shape}")

    sigma_g = np.asarray(fitdata.calerror, dtype=float).reshape(-1)  # (n_freq,)
    if sigma_g.size != w.shape[0]:
        raise ValueError(f"fitdata.calerror has size {sigma_g.size} but n_freq is {w.shape[0]}")

    if np.any(sigma_g <= 0):
        raise ValueError("All calerror (sigma_g) must be > 0 for marginalised-gain Fisher.")

    # Derivatives and fiducial model (ungained)
    dmu, mu = jacobian_finite_difference_stack(
        model=model,
        fitdata=fitdata,
        params_fid=params_fid,
        param_names=param_names,
        rel_step=rel_step,
        abs_step_floor=abs_step_floor,
        method=method,
    )
    # dmu: (n_par, n_freq, n_dpf)
    # mu : (n_freq, n_dpf)

    # --- Standard (no gain nuisance) Fisher: sum_{k,p} w * dmu_i * dmu_j
    # Vectorize:
    Wdmu = dmu * np.sqrt(w)[None, :, :]
    Jflat = Wdmu.reshape(len(param_names), -1)  # (n_par, n_freq*n_dpf)
    F0 = Jflat @ Jflat.T                        # (n_par, n_par)

    # --- Gain-marginalisation correction (rank-1 per frequency)
    # c_{k,i} = sum_p w_{kp} * mu_{kp} * dmu_{i,kp}
    c = np.einsum("kp,kp,ikp->ki", w, mu, dmu)  # (n_freq, n_par)

    # denom_k = sum_p w_{kp} * mu_{kp}^2 + 1/sigma_gk^2
    denom = np.einsum("kp,kp,kp->k", w, mu, mu) + 1.0 / (sigma_g * sigma_g)  # (n_freq,)

    # correction = sum_k outer(c_k, c_k) / denom_k
    corr = np.einsum("ki,kj,k->ij", c, c, 1.0 / denom)  # (n_par, n_par)

    F = F0 - corr

    # --- Add Gaussian prior Fisher on theta if you want (same as before)
    if global_prior is not None:
        # Minimal “NormalPrior-like” support:
        # if global_prior.bounds exists as {param:(mu,sigma)} then add 1/sigma^2 to diagonal
        if hasattr(global_prior, "bounds") and isinstance(global_prior.bounds, dict):
            for k, (mu0, sig0) in global_prior.bounds.items():
                if k in param_names and sig0 is not None and np.isfinite(sig0) and sig0 > 0:
                    i = param_names.index(k)
                    F[i, i] += 1.0 / (sig0 * sig0)

        if hasattr(global_prior, "priors"):
            for pr in getattr(global_prior, "priors"):
                if hasattr(pr, "bounds") and isinstance(pr.bounds, dict):
                    for k, (mu0, sig0) in pr.bounds.items():
                        if k in param_names and sig0 is not None and np.isfinite(sig0) and sig0 > 0:
                            i = param_names.index(k)
                            F[i, i] += 1.0 / (sig0 * sig0)

    return F


def run_fisher(
    model: Any,
    fitdata: Any,
    params_fid: Dict[str, float],
    param_names: List[str],
    global_prior: Any = None,
    rel_step: float = 1e-6,
    abs_step_floor: float = 0.0,
    method: str = "central",
    ridge: float = 0.0,
) -> FisherResult:
    """
    Full Fisher: data + (Gaussian) priors. Optionally add a small ridge to help inversion.
    """
    ivar = _get_ivar_vector(fitdata)
    J, mu = jacobian_finite_difference(
        model=model,
        fitdata=fitdata,
        params_fid=params_fid,
        param_names=param_names,
        rel_step=rel_step,
        abs_step_floor=abs_step_floor,
        method=method,
    )

    F_data = fisher_from_ivar(J, ivar)
    F_prior = fisher_from_normal_priors(global_prior, param_names)
    F_total = F_data + F_prior

    if ridge > 0:
        F_total = F_total + ridge * np.eye(F_total.shape[0])

    # Invert safely
    cov = np.linalg.pinv(F_total)
    sigma = np.sqrt(np.clip(np.diag(cov), 0, np.inf))

    denom = np.outer(sigma, sigma)
    corr = np.zeros_like(cov)
    mask = denom > 0
    corr[mask] = cov[mask] / denom[mask]

    eigvals = np.linalg.eigvalsh(F_total)

    return FisherResult(
        param_names=param_names,
        F_data=F_data,
        F_prior=F_prior,
        F_total=F_total,
        cov=cov,
        sigma=sigma,
        corr=corr,
        eigvals=eigvals,
    )
