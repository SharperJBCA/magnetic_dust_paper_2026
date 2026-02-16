import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional

# ---------- Parameterization ----------

@dataclass(frozen=True)
class ParamSpec:
    name: str
    transform: str = "identity"  # "identity" | "log" | "logit"
    # optional bounds for transformed params could go here too

class Parameterization:
    def __init__(self, specs: List[ParamSpec]):
        self.specs = specs
        self.names = [s.name for s in specs]

    def theta_to_params(self, theta: np.ndarray) -> Dict[str, float]:
        assert len(theta) == len(self.specs)
        out: Dict[str, float] = {}
        for v, spec in zip(theta, self.specs):
            if spec.transform == "identity":
                out[spec.name] = float(v)
            elif spec.transform == "log":
                out[spec.name] = float(np.exp(v))
            else:
                raise ValueError(f"Unknown transform: {spec.transform}")
        return out

# ---------- Priors ----------

class Prior:
    def logp(self, params: Dict[str, float]) -> float:
        raise NotImplementedError

class BoundsPrior(Prior):
    def __init__(self, bounds: Dict[str, Tuple[Optional[float], Optional[float]]]):
        self.bounds = bounds

    def logp(self, params: Dict[str, float]) -> float:
        for k, (lo, hi) in self.bounds.items():
            v = params[k]
            if lo is not None and v < lo:
                return -np.inf
            if hi is not None and v > hi:
                return -np.inf
        return 0.0

class NormalPrior(Prior):
    def __init__(self, name: str, mu: float, sigma: float):
        self.name, self.mu, self.sigma = name, mu, sigma

    def logp(self, params: Dict[str, float]) -> float:
        x = params[self.name]
        z = (x - self.mu) / self.sigma
        return -0.5 * (z*z + np.log(2*np.pi*self.sigma*self.sigma))

class LogUniformPrior(Prior):
    """p(x) ∝ 1/x on [lo, hi]. Requires x>0."""
    def __init__(self, name: str, lo: float, hi: float):
        self.name, self.lo, self.hi = name, lo, hi

    def logp(self, params: Dict[str, float]) -> float:
        x = params[self.name]
        if x <= 0 or x < self.lo or x > self.hi:
            return -np.inf
        return -np.log(x) - np.log(np.log(self.hi/self.lo))

class JointPrior(Prior):
    def __init__(self, priors: List[Prior]):
        self.priors = priors

    def logp(self, params: Dict[str, float]) -> float:
        lp = 0.0
        for p in self.priors:
            v = p.logp(params)
            if not np.isfinite(v):
                return -np.inf
            lp += v
        return lp

# ---------- Likelihood (diagonal Gaussian) ----------

@dataclass
class PackedData:
    d: np.ndarray
    var: np.ndarray
    # optional bookkeeping if you want diagnostics
    n: int

def pack_fitdata(fitdata, targets: List[str], stokes_order: List[str]) -> PackedData:
    ds = []
    vs = []
    for t in targets:
        m = fitdata.maps[t]  # however FitData stores per-target Map
        mask = m.mask.copy() # True for bad pixels (your convention)
        for s in stokes_order:
            y = getattr(m, s)
            vname = s + s  # "II","QQ","UU"
            vv = getattr(m, vname)

            good = (~mask) & np.isfinite(y) & np.isfinite(vv) & (vv > 0)
            ds.append(y[good])
            vs.append(vv[good])

    d = np.concatenate(ds) if ds else np.zeros(0)
    var = np.concatenate(vs) if vs else np.zeros(0)
    return PackedData(d=d, var=var, n=d.size)

def pack_prediction(prediction: Dict[str, Dict[str, np.ndarray]],
                    fitdata,
                    targets: List[str],
                    stokes_order: List[str]) -> np.ndarray:
    """
    prediction assumed like: prediction[target][stokes] -> array over region pixels
    and must line up with fitdata maps ordering + mask.
    """
    ms = []
    for t in targets:
        m = fitdata.maps[t]
        mask = m.mask.copy()
        for s in stokes_order:
            y = getattr(m, s)
            vname = s + s
            vv = getattr(m, vname)
            good = (~mask) & np.isfinite(y) & np.isfinite(vv) & (vv > 0)
            ms.append(prediction[t][s][good])
    return np.concatenate(ms) if ms else np.zeros(0)

class DiagGaussianLikelihood:
    def __init__(self, include_norm: bool = True):
        self.include_norm = include_norm

    def loglike(self, d: PackedData, m: np.ndarray) -> float:
        r = d.d - m
        chi2 = np.sum((r*r) / d.var)
        if not self.include_norm:
            return -0.5 * chi2
        return -0.5 * (chi2 + np.sum(np.log(2*np.pi*d.var)))

# ---------- Glue for emcee ----------

class LogPosterior:
    def __init__(self,
                 model,                # your Model instance
                 fitdata,
                 targets: List[str],
                 stokes_order: List[str],
                 param: Parameterization,
                 prior: Prior,
                 like: DiagGaussianLikelihood):
        self.model = model
        self.fitdata = fitdata
        self.targets = targets
        self.stokes_order = stokes_order
        self.param = param
        self.prior = prior
        self.like = like

        self.packed = pack_fitdata(fitdata, targets, stokes_order)

    def __call__(self, theta: np.ndarray) -> float:
        params = self.param.theta_to_params(theta)
        lp = self.prior.logp(params)
        if not np.isfinite(lp):
            return -np.inf

        pred = self.model.predict(self.fitdata, params)  # you already have this
        mvec = pack_prediction(pred, self.fitdata, self.targets, self.stokes_order)
        ll = self.like.loglike(self.packed, mvec)
        return lp + ll
