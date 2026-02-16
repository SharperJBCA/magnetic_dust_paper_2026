from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np 
import healpy as hp 

from ..io.raw_maps_readers import BaseReader
from ..io.maps_io import Map
from ..preprocessing.beams import smooth_maps

def scale_maps(m: Map, scale: float, variance_scale : float) -> None:
    """ 
    scale a map
    """
    bad = m.mask
    for stokes in ['I','Q','U']:
        smap = getattr(m,stokes) 
        smap[~bad] *= scale 
        setattr(m, stokes, smap) 

    for stokes in ['II','QQ','UU']:
        smap = getattr(m,stokes) 
        smap[~bad] *= variance_scale
        setattr(m, stokes, smap) 

def boundary_difference(cbass: Map, spass: Map, percentile: float) -> None:
    """ 
    find the offset between both maps 
    """
    common = (cbass.mask == False) & (spass.mask == False) 

    # cut out bright areas 
    common = common & (cbass.I < np.percentile(cbass.I[~cbass.mask], percentile))
    common = common & (spass.I < np.percentile(spass.I[~spass.mask], percentile))

    # find difference in common area 
    diff = np.nanmedian(cbass.I[common]) - np.median(spass.I[common])
    spass.I[~spass.mask] += diff 

def naive_combine(cbass: Map, spass: Map) -> Map: 

    out_map = Map.zeros_like(spass, map_id="cbass_spass_combined", stage=cbass.stage)
    
    for stokes in ['I','Q','U','II','QQ','UU']:
        smap = getattr(spass,stokes) 
        cmap = getattr(cbass,stokes) 
        omap = getattr(out_map,stokes) 

        omap[~spass.mask] = smap[~spass.mask]
        omap[spass.mask]  = cmap[spass.mask]
        setattr(out_map, stokes, omap)

    return out_map

def tapered_combine(cbass: Map, naive_map: Map, min_dec=-10,max_dec=5) -> Map:

    out_map = Map.zeros_like(cbass, map_id="cbass_spass_combined", stage=cbass.stage)

    npix = int(12*out_map.nside**2)
    pixels = np.arange(npix,dtype=int)
    gl, gb = hp.pix2ang(out_map.nside, pixels, lonlat=True) 
    rot = hp.Rotator(coord=['G','C'])
    ra,dec = rot(gl,gb,lonlat=True) 

    dec_mid = (max_dec+min_dec)/2. 
    dec_width = (max_dec-min_dec) 

    taper_map = 0.5 + 0.5*np.cos(np.pi*(dec-max_dec)/dec_width)
    taper_map[dec>max_dec] = 1.0 
    taper_map[dec<min_dec] = 0.0

    for stokes in ['I','Q','U']:
        nmap = getattr(naive_map,stokes) 
        cmap = getattr(cbass,stokes) 
        omap = getattr(out_map,stokes) 

        # taper combination at declination boundary 
        omap = cmap*taper_map + nmap*(1-taper_map) 
        setattr(out_map, stokes, omap)

    for stokes in ['II','QQ','UU']:
        nmap = getattr(naive_map,stokes) 
        cmap = getattr(cbass,stokes) 
        omap = getattr(out_map,stokes) 

        # taper combination at declination boundary 
        omap = cmap*taper_map**2 + nmap*(1-taper_map)**2
        setattr(out_map, stokes, omap)

    return out_map



def combine_cbass_spass(combine_info: Dict, cbass_reader: BaseReader, spass_reader: BaseReader) -> Map: 
    """
    Combine cbass and spass maps assuming a constant spectral index scaling.

    Steps:
    1) Scale S-PASS to C-BASS frequency (2.3 -> 4.76)
    2) Combine S-PASS and C-BASS maps, prioritise S-PASS: spass_naive 
    3) Smooth spass_naive to 1 degree resolution
    4) Combine spass_naive and cbass using a taper 

    """

    sync_beta = combine_info['sync_beta'] 
    percentile = combine_info['mask_percentile']
    target_fwhm_arcmin = combine_info['target_fwhm_arcmin']
    scale_factor = (cbass_reader.map.freq_ghz/spass_reader.map.freq_ghz)**sync_beta 
    variance_scale = 5.#np.nanmedian(cbass_reader.map.II[~cbass_reader.map.mask])/np.nanmedian(spass_reader.map.II[~spass_reader.map.mask])
    # upscale C-BASS to match S-PASS
    # this preserves the noise properties of S-PASS before we smooth
    cbass_reader.map.ud_grade(spass_reader.map.nside) 

    # now we will scale and get boundary difference
    scale_maps(spass_reader.map, scale_factor, variance_scale) 
    boundary_difference(cbass_reader.map, spass_reader.map, percentile)

    # create naive combination
    naive_map = naive_combine(cbass_reader.map, spass_reader.map)

    # now smooth this map
    smooth_maps(naive_map, spass_reader.beam_info, target_fwhm_arcmin, naive_map.nside)

    # now we combine the cbass and spass map using a taper
    final_map = tapered_combine(cbass_reader.map, naive_map) 

    return  final_map