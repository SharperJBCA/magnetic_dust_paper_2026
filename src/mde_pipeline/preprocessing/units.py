import numpy as np
import healpy as hp 
from ..io.maps_io import Map

h  = 6.62607015e-34
kb = 1.380649e-23
T_cmb = 2.725

def dBdT_cmb_to_rj(nu_ghz: float) -> float:
    """
    Conversion factor: K_CMB -> K_RJ at frequency nu.
    For small signals: T_RJ = T_CMB * (dB/dT)_CMB / (2 k nu^2 / c^2)
    """
    nu = nu_ghz * 1e9
    x = h * nu / (kb * T_cmb)
    return (x**2) * np.exp(x) / (np.exp(x) - 1.0)**2

TO_KRJ = {
    "K_RJ":   lambda nu, m=None: 1.0,
    "mK_RJ":  lambda nu, m=None: 1e-3,
    "uK_RJ":  lambda nu, m=None: 1e-6,
    "K_CMB":  lambda nu, m=None: dBdT_cmb_to_rj(nu),
    "mK_CMB": lambda nu, m=None: 1e-3 * dBdT_cmb_to_rj(nu),
    "uK_CMB": lambda nu, m=None: 1e-6 * dBdT_cmb_to_rj(nu),
}

def valid_mask(m) -> np.ndarray:
    if isinstance(m.mask, np.ndarray) and m.mask.size == m.I.size:
        return m.mask.astype(bool)

    v = np.isnan(m.I)
    v &= (m.I == hp.UNSEEN)
    return v

def unit_convert(m, beam_info, to: str = "K_RJ") -> None:
    from_unit = m.unit
    if not from_unit:
        raise ValueError(f"{m.map_id}: map.unit is empty; cannot convert")

    if "unit_native" not in m.meta:
        m.meta["unit_native"] = from_unit

    if from_unit == to:
        return

    if from_unit not in TO_KRJ:
        raise KeyError(f"{m.map_id}: unsupported unit '{from_unit}'")
    if to not in TO_KRJ:
        raise KeyError(f"{m.map_id}: unsupported target unit '{to}'")

    nu = float(m.freq_ghz)
    if not np.isfinite(nu):
        raise ValueError(f"{m.map_id}: freq_ghz is not set; cannot convert units")

    f_from = TO_KRJ[from_unit](nu, m)
    f_to   = TO_KRJ[to](nu, m)
    factor = f_from / f_to

    v = valid_mask(m)

    for s in ["I", "Q", "U", "II", "QQ", "UU"]:
        arr = getattr(m, s)
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            continue

        if s in ["II", "QQ", "UU"]:
            arr[~v] *= factor**2  
        else:
            arr[~v] *= factor

        setattr(m, s, arr)

    m.meta.setdefault("unit_history", []).append({"from": from_unit, "to": to, "factor": factor, "nu_ghz": nu})
    m.unit = to


def fix_pol_convention(m, beam_info, to_pol_convention='COSMO'):
    if m.pol_convention != to_pol_convention:
        if "pol_convention_native" not in m.meta:
            m.meta["pol_convention_native"] = m.pol_convention

        v = valid_mask(m)
        m.U[~v] *= -1
        m.pol_convention = to_pol_convention


def dec_mask(m, beam_info, min_dec=-90, max_dec=90):    
    for k in ['I','Q','U','II','QQ','UU','mask']:
        if hasattr(m,k):
            tmp = getattr(m,k).copy()
            npix = tmp.size 
            nside = hp.npix2nside(npix) 
            pixels = np.arange(npix,dtype=int)
            rot = hp.Rotator(coord=['G','C']) 
            gl,gb = hp.pix2ang(nside,pixels,lonlat=True)
            ra,dec = rot(gl,gb,lonlat=True)
            mask = (dec < min_dec) 
            if 'k' == 'mask':
                tmp[mask] = True
            else:
                tmp[mask] = hp.UNSEEN 
            mask = (dec > max_dec) 
            if 'k' == 'mask':
                tmp[mask] = True
            else:
                tmp[mask] = hp.UNSEEN 
            setattr(m,k, tmp)
    
class subtract_cmb: 

    def __init__(self):
        self._cmb_cache = {} 

    def __call__(self, m : Map, beam_info,cmb_map): 
        if m.nside not in self._cmb_cache:
            cmb_i, cmb_q, cmb_u = hp.read_map(cmb_map, field=[0, 1, 2])
            cmb_i = hp.ud_grade(cmb_i, m.nside)
            cmb_q = hp.ud_grade(cmb_q, m.nside)
            cmb_u = hp.ud_grade(cmb_u, m.nside)
            self._cmb_cache[m.nside] = {'I': cmb_i, 'Q': cmb_q, 'U': cmb_u}
        cmb = self._cmb_cache[m.nside] 

        for stokes in ['I','Q','U']:
            s = getattr(m,stokes)
            if s.size > 0: 
                s[~m.mask] -= cmb[stokes][~m.mask] * dBdT_cmb_to_rj(m.freq_ghz)



