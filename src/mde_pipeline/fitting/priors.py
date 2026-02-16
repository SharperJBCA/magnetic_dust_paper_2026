from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional, Any

class Prior:
    def logp(self, params: Dict[str, float]) -> float:
        raise NotImplementedError

    def __call__(self, params: Dict[str, float]) -> float:
        return self.logp(params)

    def __add__(self, other):
        if other is None:
            return self
        if isinstance(self, JointPrior) and isinstance(other, JointPrior):
            return JointPrior(self.priors + other.priors)
        if isinstance(self, JointPrior):
            return JointPrior(self.priors + [other])
        if isinstance(other, JointPrior):
            return JointPrior([self] + other.priors)
        return JointPrior([self, other])

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
    def __init__(self, bounds: Dict[str, Tuple[Optional[float], Optional[float]]]):
        self.bounds = bounds

    def logp(self, params: Dict[str, float]) -> float:
        logp = 0.0
        for k, (mu,sigma) in self.bounds.items():
            v = params[k]
            z = (v - mu) / sigma
            logp += -0.5 * (z*z + np.log(2*np.pi*sigma*sigma))
        return logp

class LogUniformPrior(Prior):
    """p(x) ∝ 1/x on [lo, hi]. Requires x>0."""
    def __init__(self, name: str, lo: float, hi: float):
        self.name, self.lo, self.hi = name, lo, hi

    def logp(self, params: Dict[str, float]) -> float:
        x = params[self.name]
        if x <= 0 or x < self.lo or x > self.hi:
            return -np.inf
        return -np.log(x) - np.log(np.log(self.hi/self.lo))


PRIOR_REGISTRY = {
    "bounds": BoundsPrior,
    "normal": NormalPrior,
}

def build_prior(prior_list: Optional[List[Dict[str, Any]]]) -> Optional[Prior]:
    """
    prior_list: YAML 'priors' list
    Returns a Prior (possibly JointPrior via __add__) or None.
    """
    if not prior_list:
        return None

    out: Optional[Prior] = None
    for item in prior_list:
        ptype = item["type"].lower()
        params = item.get("params", {})

        if ptype == "bounds":
            # expect {"A": [lo, hi], "beta": [lo, hi]}
            bounds = {k: (v[0], v[1]) for k, v in params.items()}
            p = BoundsPrior(bounds)

        elif ptype == "normal":
            # expect {"beta": [mu, sigma], ...}
            p = NormalPrior(params)

        else:
            raise ValueError(f"Unknown prior type '{ptype}'. Supported: {list(PRIOR_REGISTRY)}")

        out = p if out is None else (out + p)

    return out
