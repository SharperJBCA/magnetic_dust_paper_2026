from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

from ..utils.logging import get_logger
from ..utils.config import load_yaml
from ..io import raw_maps_readers 
from ..io.maps_io import MapIO, Map  

from .units import fix_pol_convention, unit_convert, dec_mask, subtract_cmb
from .beams import smooth_maps
from ..qc.qc_plotting import qc_plot_map

log = get_logger(__name__)

OPS = {
    "fix_pol_convention": fix_pol_convention,
    "unit_convert": unit_convert, 
    "smooth_maps": smooth_maps,
    "dec_mask":dec_mask,
    "subtract_cmb":subtract_cmb()
}

def run_preprocess(
    instruments_yaml: Path,
    preprocess_yaml: Path,
    out_h5: Path,
    overwrite: bool = False,
    dry_run: bool = False,
    only: Optional[list[str]] = None,
) -> None:
    log.info("Loading instruments config: %s", instruments_yaml)
    instruments_info: Dict[str, Any] = load_yaml(instruments_yaml)

    log.info("Loading preprocess config: %s", preprocess_yaml)
    preprocess_info: Dict[str, Any] = load_yaml(preprocess_yaml)

    # Decide which map_ids to process
    map_ids = list(instruments_info.keys())
    if only is not None:
        map_ids = [m for m in map_ids if m in set(only)]

    if not map_ids:
        log.warning("No map_ids selected. (only=%s)", only)
        return

    # Make sure output path exists
    if not dry_run:
        out_h5.parent.mkdir(parents=True, exist_ok=True)

    # Your MapIO should probably accept a direct file path; easiest is:
    map_io = MapIO(data_path=str(out_h5.parent), filename=out_h5.name)

    log.info("Preprocessing %d map(s) -> %s", len(map_ids), out_h5)

    for map_id in map_ids:
        meta = instruments_info[map_id]
        steps = preprocess_info.get(map_id, preprocess_info.get("_default", []))

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
            log.info("[%s] (dry-run) would execute steps: %s", map_id, steps)
            continue

        reader = reader_cls(map_id,meta)

        # # --- apply preprocess ops (you’ll implement these) ---
        for step_info in steps:
            op = step_info['op'] 
            kwargs = {k:v for k,v in step_info.items() if k != 'op'}
            OPS[op](reader.map, reader.beam_info, **kwargs) 

        log.info("[%s] Writing to %s group=%s", map_id, out_h5, map_id)
        tag = str(out_h5.parent.name)
        reader.map.stage = 'processed'
        map_io.write_map(reader.map) 

        if False:
            qc_plot_map(
                reader.map,
                preprocess_info.get("qc",{}),
                out_dir=Path(f"products/qc/preprocess/{tag}") / map_id,
            )

    log.info("Preprocess complete.")
