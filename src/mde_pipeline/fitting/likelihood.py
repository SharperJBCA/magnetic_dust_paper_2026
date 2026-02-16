import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Optional

from ..emission.components import COMPONENTS
from .data_types import ComponentSpec


class Likelihood:

    def __init__(self, components: List[ComponentSpec], global_prior=None):
        self.components = components
        self.global_prior = global_prior
        # instantiate emission components once
        self._comps = []
        for spec in self.components:
            cls = COMPONENTS[spec.cls_name]
            self._comps.append((spec, cls(name=spec.name)))

    def chi2(self, params, fitdata, model):
        prediction = model.predict(fitdata, params)

        chi2 = np.sum((fitdata.data - prediction)**2 * fitdata.ivar)
        return chi2 #np.sum((np.log(fitdata.data) - np.log(prediction))**2)#chi2  
    
    def __call__(self, params, fitdata, model):
        
        lnp = 0.0

        #if self.global_prior is not None: 
        #    lp = self.global_prior(params) 
        #    if not np.isfinite(lp):
        #        return -np.inf 
        #    lnp += lp 

        for spec, comp in self._comps:
            if spec.priors is None: 
                continue
            comp_params = dict(spec.fixed_params)
            comp_params.update({arg: params[name] for arg, name in spec.params_map.items() if name in params})
            lp = spec.priors(comp_params)
            if not np.isfinite(lp):
                return -np.inf 
            lnp += lp 
            
        prediction = model.predict(fitdata, params)
        d = fitdata.data
        w = fitdata.ivar

        sigma_g = fitdata.calerror        
        mu_g = 1.0                            # or fitdata.gain_mu

        # A_k, B_k per frequency (k = freq index)
        A = np.sum(w * prediction**2, axis=1) + 1.0 / sigma_g**2
        B = np.sum(w * prediction * d,  axis=1) + mu_g / sigma_g**2

        lngain_marg = 0.5 * (B**2 / A) - 0.5 * np.log(A)

        lnlike = (
            -0.5 * np.sum(w * d**2)
            + np.sum(lngain_marg)
            + 0.5 * np.sum(np.log(w))
            - 0.5 * d.size * np.log(2.0 * np.pi)
        )
        if np.isnan(lnlike):
            return -np.inf 
        return lnlike + lnp