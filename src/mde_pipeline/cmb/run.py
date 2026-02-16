from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import os 

import healpy as hp 

from ..utils.logging import get_logger
from ..utils.config import load_yaml
from ..io import raw_maps_readers 
from ..io.maps_io import MapIO, Map  

from ..qc.qc_plotting import qc_plot_map
from ..preprocessing.beams import smooth_maps

log = get_logger(__name__)

def run_smooth_cmb(
    instruments_yaml: Path,
    target_fwhm_arcmin: float,
    out_fits: Path,
    overwrite: bool = True,
    dry_run: bool = False
) -> None:
    log.info("Loading instruments config: %s", instruments_yaml)
    instruments_info: Dict[str, Any] = load_yaml(instruments_yaml)

    for map_id in ['cmb_commander']:
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
            log.info("[%s] (dry-run) for smooth cmb", map_id)
            continue

        reader = reader_cls(map_id,meta)

    if dry_run: 
        return 

    smooth_maps(reader.map, reader.beam_info, target_fwhm_arcmin, reader.map.nside)
    os.makedirs(out_fits.parent, exist_ok=True)
    hp.write_map(out_fits,[reader.map.I,reader.map.Q,reader.map.U],overwrite=overwrite)