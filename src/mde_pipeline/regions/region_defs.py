from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import os 
import numpy as np
import healpy as hp 

from ..utils.logging import get_logger
from ..utils.config import load_yaml
from ..io import raw_maps_readers 
from ..io.maps_io import MapIO, Map  


def percentile_threshold(mapio : MapIO, masks : Dict, map_id: str, stokes:str, percentile : float,  max_lat_threshold : float, min_lat_threshold : float, max_lat : float = None, min_lat : float = None,lon_range:List=None, inner_lon=False):
    m = mapio.read_map(map_id)

    I_map = getattr(m,stokes) 
    nside = hp.npix2nside(I_map.size) 
    pixels = np.arange(I_map.size,dtype=int) 

    lower_threshold = np.percentile(I_map[~m.mask],percentile)

    gl,gb = hp.pix2ang(nside, pixels, lonlat=True) 
    gl = np.mod(gl,360)
    mask = (gb < max_lat_threshold) & (gb > min_lat_threshold) & (I_map > lower_threshold) & (m.mask == False)

    if isinstance(max_lat, float) and isinstance(min_lat, float):
        mask |= ((gb < max_lat) & (gb > min_lat))
    if isinstance(lon_range, list):
        lon1 = lon_range[0]
        lon2 = lon_range[1] 
        if inner_lon: # wrapping around 0 
            mask_lon = ((gl >= 0) & (gl < lon1)) | ((gl <= 360) & (gl > lon2))
        else: # normal range 
            mask_lon = (gl >= lon1) & (gl < lon2) 
        mask &= mask_lon

    return mask 

def regions_minus_masks(mapio :MapIO, masks : Dict, mask_names:List, lon_range:List, lat_range:List):

    mask = np.zeros_like(list(masks.values())[0])
    for mask_name in mask_names: 
        mask_region = masks[mask_name]
        mask |= mask_region 

    nside = hp.npix2nside(mask.size) 
    pixels = np.arange(mask.size,dtype=int) 
    gl,gb = hp.pix2ang(nside, pixels, lonlat=True) 
    gl = np.mod(gl,360)

    lon1 = lon_range[0]
    lon2 = lon_range[1] 
    #if lon1 < lon2: # wrapping around 0 
    #    mask_lon = ((gl >= 0) & (gl < lon1)) | ((gl <= 360) & (gl > lon2))
    #else: # normal range 
    mask_lon = (gl >= lon1) & (gl < lon2) 
    mask_lat = (gb >= lat_range[0]) & (gb < lat_range[1]) 

    out_mask = (mask_lon & mask_lat) & (mask == False)


    return out_mask