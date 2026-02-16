from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import os 

import numpy as np
import healpy as hp 
from matplotlib import pyplot

from ..utils.logging import get_logger
from ..utils.config import load_yaml
from ..io import raw_maps_readers 
from ..io.maps_io import MapIO, Map  

from ..io.regions_io import RegionsIO
from .region_defs import percentile_threshold,regions_minus_masks
from ..qc.qc_plotting import qc_plot_map

log = get_logger(__name__)

REGISTER = {
    'percentile_threshold':percentile_threshold,
    'regions_minus_masks':regions_minus_masks
}

def run_regions(
    regions_yaml: Path,
    processed_h5: Path, # processed maps 
    out_h5: Path, # output regions file 
    overwrite: bool = True,
    dry_run: bool = False
) -> None:
    log.info("Loading regions config: %s", regions_yaml)
    regions_info: Dict[str, Any] = load_yaml(regions_yaml)['regions']

    group = str(processed_h5.parent.name) # v001/etc...
    mapio = MapIO(processed_h5.parent, processed_h5.name, group)

    masks = {}
    for mask_name, mask_info in regions_info['masks'].items():
        func = REGISTER[mask_info["type"]]
        kwargs = mask_info["kwargs"]
        masks[mask_name] = func(mapio, masks, **kwargs)

    npix = list(masks.values())[0].size
    nside = hp.npix2nside(npix)
    region_map = np.zeros(npix,dtype=int)

    mask_count = 1 # start at id 1
    for mask_name, mask in masks.items(): 
        region_map[mask] = mask_count 
        mask_count += 1

    if not os.path.exists(out_h5.parent):
        os.makedirs(out_h5.parent, exist_ok=True)
    regions_io = RegionsIO(out_h5, group)
    regions_io.write_regions(regions_info['group_name'],region_map, masks, {'nside':nside})

    if regions_info['qc']['enabled']:
        out_path = Path(f"products/qc/regions/{group}") / regions_info['group_name'] / "region_map.png"
        if not os.path.exists(out_path.parent):
            os.makedirs(out_path.parent, exist_ok=True)

        hp.mollview(region_map, title=regions_info['group_name'])
        hp.graticule() 
        pyplot.savefig(out_path)
        pyplot.close()