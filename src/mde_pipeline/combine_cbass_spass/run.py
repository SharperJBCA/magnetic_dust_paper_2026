from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import os 

import healpy as hp 

from ..utils.logging import get_logger
from ..utils.config import load_yaml
from ..io import raw_maps_readers 
from ..io.maps_io import MapIO, Map  

from .combine import combine_cbass_spass
from ..qc.qc_plotting import qc_plot_map

log = get_logger(__name__)

def run_combine_cbass_spass(
    combine_yaml: Path,
    instruments_yaml: Path,
    out_fits: Path,
    overwrite: bool = True,
    dry_run: bool = False
) -> None:
    log.info("Loading instruments config: %s", instruments_yaml)
    instruments_info: Dict[str, Any] = load_yaml(instruments_yaml)

    log.info("Loading C-BASS/S-PASS combine config: %s", combine_yaml)
    combine_info: Dict[str, Any] = load_yaml(combine_yaml)

    spass_id = combine_info['spass_id']
    cbass_id = combine_info['cbass_id'] 
    # Read in S-PASS and C-BASS
    readers = {}
    for map_id in [spass_id,cbass_id]:
        meta = instruments_info[map_id]
        read_class = meta.get("read_class")
        if not read_class:
            log.error("[%s] missing read_class", map_id)
            continue

        if not hasattr(raw_maps_readers, read_class):
            log.error("[%s] reader class not found: %s", map_id, read_class)
            continue

        reader_cls = getattr(raw_maps_readers, read_class)

        log.info("[%s] Reading raw map using %s", map_id, read_class)
        if dry_run:
            log.info("[%s] (dry-run) for s-pass c-bass combine", map_id)
            continue

        readers[map_id] = reader_cls(map_id,meta)

    if dry_run: 
        return 
    final_map = combine_cbass_spass(combine_info, readers[cbass_id], readers[spass_id])

    os.makedirs(out_fits.parent, exist_ok=True)
    hp.write_map(out_fits,[final_map.I,final_map.Q,final_map.U,final_map.II,final_map.QQ,final_map.UU],overwrite=overwrite)