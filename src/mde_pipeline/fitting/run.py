from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np 

from ..utils.logging import get_logger
from ..utils.config import load_yaml
from ..io import raw_maps_readers 
from ..io.maps_io import MapIO, Map  

from ..io.regions_io import RegionsIO
from ..templates.templates import load_templates_config
from ..emission.components import COMPONENTS
from ..qc.qc_plotting import qc_plot_map

from .data_types import Model, ComponentSpec, FitData, build_components_from_yaml
from .priors import NormalPrior, BoundsPrior
from .likelihood import Likelihood
from .emcee_runner import run_emcee_region,ParamVector, load_samples_npz
from .fit_output import write_region_products,write_fisher_region_products

from .fisher import run_fisher,fisher_gain_marginalized

log = get_logger(__name__)

from pathlib import Path

def regen_region_products_from_npz(
    samples_npz: Path,
    out_run_dir: Path,
    run_name_tag: str,
    maps: dict,            # your loaded maps dict (map_name -> Map)
    templates: dict,       # your loaded templates dict (template_name -> Template)
    components: list,
    Model,
    FitData,
    stokes_order=("I","Q","U"),
    include_lnL_norm=False,
    posterior_predictive_draws=200,
    nside=None,
    make_healpix_maps=False,
):
    result, meta = load_samples_npz(samples_npz)
    region_name = meta["region_name"]
    pixels = meta["pixels"]

    # rebuild fitdata/model using pixels from npz
    fitdata = FitData.create_from_dict(
        {map_name: v.slice_map(v, pixels) for map_name, v in maps.items()},
        list(stokes_order),
        pixels,
    )
    region_templates = {tname: v.slice_template(v, pixels) for tname, v in templates.items()}
    model = Model(components, region_templates, stokes_order=list(stokes_order))

    pv = ParamVector(result.param_names)

    # call the same writer
    return write_region_products(
        out_run_dir=out_run_dir,
        run_name_tag=run_name_tag,
        region_name=region_name,
        fitdata=fitdata,
        model=model,
        result=result,          # EmceeResultLite has the same fields we use
        param_vector=pv,
        include_lnL_norm=include_lnL_norm,
        posterior_predictive_draws=posterior_predictive_draws,
        nside=nside,
        make_healpix_maps=make_healpix_maps,
    )


def run_fit(
    fitter_yaml: Path,
    data_yaml: Path,
    templates_yaml: Path,
    regions_h5: Path,
    processed_h5: Path, # can be simulations or real data
    out_dir: Path,
    run_name: str,
    region_ids: Optional[List[str]],
    overwrite: bool,
    dry_run: bool,
) -> None:
    
    fitter_info = load_yaml(fitter_yaml)["fitter"]
    tag = 'v001' # for regions
    sims_h5 = Path(fitter_info['sims_h5'])
    sim_tag =  fitter_info['sims_tag']
    regions = RegionsIO(regions_h5, tag).load_regions('gal_plus_high_1')
    processed_h5 = Path(fitter_info["processed_h5"])
    templates = load_templates_config(templates_yaml, processed_h5) 

    mapsio = MapIO(data_path=str(sims_h5.parent), filename=sims_h5.name) 
    out_dir = Path(fitter_info['out_dir']) / sim_tag 
    maps = {}
    for map_name in fitter_info["targets"]:
        maps[map_name] = mapsio.read_map(map_name)
    if not region_ids:
        region_ids = regions.region_names


    components, param0, widths0, global_prior, gain_param_names = build_components_from_yaml(fitter_info)
    #global_prior = None
    lnlike_obj = Likelihood(components, None)

    run_mcmc = True
    region_templates = {}
    for region_name in region_ids:
        pixels = regions.get_pixels(region_name)

        fitdata = FitData.create_from_dict( {map_name:v.slice_map(v,pixels) for map_name, v in maps.items()},
                                           ["I","Q","U"],
                                           pixels)
        idx = 0
        for j, map_name in enumerate(fitdata.map_names):
            if f"cal_{map_name}" in param0:
                fitdata.calerror[j] = global_prior.priors[idx].bounds[f"cal_{map_name}"][1]
                idx += 1

        region_templates[region_name] = {template_name:v.slice_template(v,pixels) for template_name, v in templates.items()}
        model = Model(components, region_templates[region_name], stokes_order=['I','Q','U'])

        # ---- Fisher sanity check (before MCMC) ----
        # Build fiducial params dict including gains 

        # fiducial params = param0 (your science params in sampling space)
        params_fid = dict(param0)
        param_names = list(params_fid.keys())

        #global_prior.priors[-1].bounds = [1.0,0.1]
        F = fisher_gain_marginalized(
            model=model,
            fitdata=fitdata,
            params_fid=params_fid,
            param_names=param_names,
            global_prior=global_prior,   # optional
            rel_step=1e-6,
            method="central",
        )

        cov = np.linalg.pinv(F)
        sig = np.sqrt(np.clip(np.diag(cov), 0, np.inf))

        log.info(f"[FISHER gain-marg] region={region_name}")
        fisher_result = {}
        for n, s in zip(param_names, sig):
            log.info(f"  sigma({n}) = {s:.4g}")

            fisher_result[n] = s

        pv = ParamVector(param_names)
        summary = write_fisher_region_products(
            out_run_dir=out_dir,
            run_name_tag=sim_tag,
            region_name=region_name,
            fitdata=fitdata,
            model=model,
            fisher_result=fisher_result,
            fisher_cov=cov,
            params0=param0,
            param_vector=pv
        )

        import sys
        sys.exit()

        if run_mcmc:
            result = run_emcee_region(
                lnpost_obj=lnlike_obj,
                fitdata=fitdata,
                model=model,
                params0=param0,
                widths0=widths0,
                extra_param_names=gain_param_names,
                components=components,
                burn_steps=500,
                burn_stages=1,
                prod_steps=2500,
                prod_burnin=1500,
                thin=1,
                nwalkers=64,
                prune_every=250,
                seed=0,
            )

            pv = ParamVector(result.param_names)

            summary = write_region_products(
                out_run_dir=out_dir,
                run_name_tag=sim_tag,
                region_name=region_name,
                fitdata=fitdata,
                model=model,
                result=result,
                param_vector=pv,
                include_lnL_norm=False,
                posterior_predictive_draws=200,  
                nside=None,                      
                make_healpix_maps=False,       
            )

        else:
            samples = out_dir / "regions" / region_name / "samples.npz"

            regen_region_products_from_npz(
                samples_npz=samples,
                out_run_dir=out_dir,
                run_name_tag=sim_tag,
                maps=maps,
                templates=templates,
                components=components,
                Model=Model,
                FitData=FitData,
                posterior_predictive_draws=500,
                make_healpix_maps=True,
                nside=16,
            )
        break