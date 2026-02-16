import numpy as np 
from typing import Dict, List, Any
from dataclasses import dataclass 
from pathlib import Path 
import json 
from ..emission.components import COMPONENTS, EmissionComponent
from ..io.maps_io import Map
from .priors import Prior, build_prior


@dataclass
class ComponentSpec:
    name: str
    cls_name: str
    template_name: str
    params_map: Dict[str, str]      
    stokes: List[str]              
    fixed_params: Dict[str,float]
    priors: Prior

@dataclass 
class FitData: # computed for each region 
    data: np.ndarray[float,float]  # nfreq by npixels*nstokes
    ivar: np.ndarray[float,float]
    stokes: list[str] # I/Q/U 
    frequencies_ghz: np.ndarray[float] 
    pixels: np.ndarray
    calerror: float 
    map_names: list[str]


    @staticmethod
    def create_from_dict(data_dict, stokes, pixels):
        stokes = stokes
        pixels = pixels
        nstokes = len(stokes)
        cov_for = {"I":"II", "Q":"QQ", "U":"UU"}
        for k,v in data_dict.items():
            npixels = v.I.size 
            break
        nmaps = len(data_dict.keys()) 
        data = np.zeros((nmaps, npixels*nstokes))
        ivar = np.zeros((nmaps, npixels*nstokes))
        frequencies_ghz = np.zeros(nmaps)
        calerror = np.zeros(nmaps)
        map_names = []
        for i,(k,v) in enumerate(data_dict.items()):
            for istokes, s in enumerate(stokes):
                data[i,istokes*npixels:(istokes+1)*npixels] = getattr(v,s)
                cov_id = cov_for[s] 
                ivar[i,istokes*npixels:(istokes+1)*npixels] = 1./getattr(v,cov_id)
            frequencies_ghz[i] = getattr(v,'freq_ghz') 
            calerror[i] = getattr(v,'calerr')
            map_names.append(v.map_id)

        return FitData(
            data = data, 
            ivar = ivar, 
            stokes = stokes, 
            frequencies_ghz = frequencies_ghz, 
            pixels = pixels,
            calerror=calerror,
            map_names=map_names
        )


def build_components_from_yaml(cfg: Dict[str, Any]) -> List[ComponentSpec]:
    comps: List[ComponentSpec] = []
    param0 = {}
    widths0 = {}
    global_prior: Prior = None 

    for c in cfg.get("components", []):
        priors = build_prior(c.get("priors", None))

        comps.append(
            ComponentSpec(
                name=c["name"],
                cls_name=c["cls_name"],
                template_name=c["template_name"],
                stokes=list(c.get("stokes", ["I", "Q", "U"])),
                params_map=dict(c.get("params_map", {})),
                fixed_params=dict(c.get("fixed_params", {})),
                priors=priors,
            )
        )
        for class_param, global_param in comps[-1].params_map.items():
            param0[global_param] = c["init"][class_param][0]
            widths0[global_param] = c["init"][class_param][1]

    # build gain parameters 
    # read in the gains
    gain_param_names = []
    if True:
        gain_filename = Path(cfg['sims_h5']).parent / "gains.json" 
        with open(gain_filename,'r') as gain_file:
            gains = json.load(gain_file) 
        
        # 
        for g in cfg.get("gains",[]):
            gparam = g["param"] 
            target = gparam.split('cal_')[1]
            param0[gparam] = float(gains.get(target,1.0))
            gain_param_names.append(gparam)
            mu0,sig0 = g['priors'][0]['params'][gparam]
            g['priors'][0]['params'][gparam] = [float(gains.get(target,1.0)),sig0]
            gp = build_prior(g.get("priors",None))
            if gp is not None: 
                global_prior = gp if global_prior is None else (global_prior + gp)


    return comps, param0, widths0, global_prior, gain_param_names

class Model:
    def __init__(self, components: List[ComponentSpec], templates: Dict[str, object], stokes_order: List[str]):
        self.components = components
        self.templates = templates
        self.stokes_order = stokes_order

        # instantiate emission components once
        self._comps = []
        for spec in self.components:
            cls = COMPONENTS[spec.cls_name]
            self._comps.append((spec, cls(name=spec.name)))

        # map stokes label -> axis index
        self._sidx = {s: i for i, s in enumerate(stokes_order)}

    def predict(self,  fitdata: FitData, params: Dict[str, float]) -> np.ndarray:

        nmaps = fitdata.frequencies_ghz.size
        nstokes = len(fitdata.stokes)
        npix = fitdata.pixels.size 

        pred = np.zeros((nmaps,nstokes*npix))


        for spec, comp in self._comps:

            comp_params = dict(spec.fixed_params)
            comp_params.update({arg: params[name] for arg, name in spec.params_map.items() if name in params})

            for i, nu in enumerate(fitdata.frequencies_ghz):
                out = comp.evaluate(nu_ghz=nu, 
                                    T=self.templates[spec.template_name], 
                                    params=comp_params)
                for s in spec.stokes:
                    if s not in out:
                        continue
                    pred[i, self._sidx[s]*npix:(self._sidx[s]+1)*npix] += out[s]

        # Now add the gain parameters - we are going to do this in the likelihood
                    # we will marginalize out the gains, not sample them
        for j, map_name in enumerate(fitdata.map_names):
           g = params.get(f"cal_{map_name}",1.0) 
           pred[j,:] *= g 

        return pred