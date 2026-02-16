from __future__ import annotations

from pathlib import Path
import numpy as np

from ..utils.logging import get_logger
from ..utils.config import load_yaml
from ..io.maps_io import MapIO, Map

from ..templates.templates import load_templates_config
from ..emission.components import COMPONENTS
from ..qc.qc_plotting import qc_plot_map

from .qc_simulations import plot_spectrum
import json 

log = get_logger(__name__)

def make_empty_like(target: Map, stage="sim") -> Map:
    m = Map(map_id=target.map_id, stage=stage)
    # copy attrs
    for k in target.attribute_fields:
        setattr(m, k, getattr(target, k))
    m.beam = target.beam
    m.mask = target.mask.copy() if target.mask.size else target.mask
    # allocate arrays
    m.I = np.zeros_like(target.I)
    m.Q = np.zeros_like(target.Q) if target.Q.size else np.empty(0, np.float32)
    m.U = np.zeros_like(target.U) if target.U.size else np.empty(0, np.float32)
    # variances: either copy (if you treat them as “known noise model”) or zero them
    m.II = target.II.copy() if target.II.size else np.empty(0, np.float32)
    m.QQ = target.QQ.copy() if target.QQ.size else np.empty(0, np.float32)
    m.UU = target.UU.copy() if target.UU.size else np.empty(0, np.float32)
    m.meta = dict(target.meta)  # shallow copy ok
    m.meta["simulated"] = True
    return m

def validate_template_target(T, target: Map):
    if T.m.nside != target.nside:
        raise ValueError(f"Template {T.name} nside={T.m.nside} != target {target.map_id} nside={target.nside}")
    if T.m.coord != target.coord:
        raise ValueError(f"Template {T.name} coord={T.m.coord} != target {target.map_id} coord={target.coord}")
    if T.m.I.size != target.I.size:
        raise ValueError("Template/target pixel size mismatch")
    
def add_noise(m: Map):
    good = (m.mask == 0) if m.mask.size else slice(None)
    if m.II.size:
        m.I[good] += np.random.normal(scale=np.sqrt(m.II[good]))
    if m.has_pol and m.QQ.size and m.UU.size:
        m.Q[good] += np.random.normal(scale=np.sqrt(m.QQ[good]))
        m.U[good] += np.random.normal(scale=np.sqrt(m.UU[good]))

def add_gain(m: Map):
    good = (m.mask == 0) if m.mask.size else slice(None)
    gain = np.random.normal(loc=1.0,scale=m.calerr)
    if m.II.size:
        m.I[good] *= gain
    if m.has_pol and m.QQ.size and m.UU.size:
        m.Q[good] *= gain
        m.U[good] *= gain

    return gain 

def run_simulations(
    sims_yaml: Path,
    overwrite: bool,
    dry_run: bool,
) -> None:
    cfg = load_yaml(sims_yaml)['simulations']
    processed_h5 = Path(cfg["processed_h5"])
    out_h5 = Path(cfg["out_h5"])
    out_gain_file = out_h5.parent / "gains.json" 

    templates = load_templates_config(cfg['templates'], processed_h5)
    mapio = MapIO(processed_h5.parent, processed_h5.name)
    mapio_out = MapIO(out_h5.parent, out_h5.name)
    sim_tag = out_h5.parent.name # just for some plotting stuff 

    components = {}

    simulated_maps = {}
    gains = {}
    for target_name in cfg['targets']:
        target = mapio.read_map(target_name, )
        simulated_maps[target_name] = make_empty_like(target)
        for comp_cfg in cfg["components"]:
            cls = COMPONENTS[comp_cfg["class"]]
            comp = cls(name=comp_cfg["name"])
            T = templates[comp_cfg["template"]] 
            params = comp_cfg.get("params", {})
            validate_template_target(T, target)

            if comp_cfg["name"] not in components:
                components[comp_cfg["name"]]={
                    "params":params,
                    "template":T, 
                    "component": comp
                }

            pred = comp.evaluate(nu_ghz=target.freq_ghz, 
                                 T=T, 
                                 params=params, 
                                 ctx={"target": target.map_id})
            
            for stokes, stoke_map in pred.items():
                s = getattr(simulated_maps[target_name],stokes)
                setattr(simulated_maps[target_name],stokes, s+stoke_map)
        
            if cfg['gain']['enabled']:
                gains[target_name] = add_gain(simulated_maps[target_name])
            if cfg['noise']['enabled']:
                add_noise(simulated_maps[target_name])

            mapio_out.write_map(simulated_maps[target_name])

            if cfg['qc']['enabled']:
                qc_plot_map(
                    simulated_maps[target_name],
                    cfg.get("qc",{}),
                    out_dir=Path(f"products/qc/simulations/{sim_tag}") / target.map_id,
                )

    with open(out_gain_file,'w') as outfile:
        json.dump(gains, outfile, sort_keys=False) 
    plot_spectrum(components,str(out_h5.parent))
