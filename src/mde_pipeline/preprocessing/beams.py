import numpy as np 
import healpy as hp 
from ..io.maps_io import BeamInfo, Map

from .units import valid_mask

def square_beam(Bl, nside, lmax, theta_bins=1000):
    """
    Square the beam transfer function in sky frame and return it. 

    Parameters
    ----------
    Bl : array-like
        The beam transfer function.
    lmax : int
    
    """
    PIXAREA = hp.nside2pixarea(nside)

    theta = np.linspace(0, np.pi, theta_bins)
    btheta = hp.bl2beam(Bl, theta)**2

    Bl2 = hp.beam2bl(btheta, theta, lmax=lmax) 
    Bl2 /= Bl2[0]
    return Bl2 #* PIXAREA 

def smooth_map(maps, bl, lmax=None, nside_out=None):
    """
    Smooth either a temperature-only map or an IQU set using an l-space filter `bl`,
    then (optionally) downgrade with ud_grade.

    Parameters
    ----------
    maps : array-like or sequence
        Either:
          - I (np.ndarray)
          - [I] (sequence length 1)
          - [I, Q, U] (sequence length 3)
    bl : array-like
        l-space filter kernel to multiply alms by (K_l). Must be defined from l=0..lmax.
        For polarization, l=0,1 are set to zero for spin-2.
    lmax : int, optional
        Max multipole to use (will be clipped to min(3*nside-1, len(bl)-1)).
    nside_out : int, optional
        Output nside. If None, keep native nside.

    Returns
    -------
    If temperature-only input: dI
    If IQU input: (dI, dQ, dU)
    """

    # ---- normalize inputs ----
    if isinstance(maps, (list, tuple)):
        if len(maps) == 1:
            I = np.asarray(maps[0])
            Q = U = None
        elif len(maps) == 3:
            I, Q, U = (np.asarray(maps[0]), np.asarray(maps[1]), np.asarray(maps[2]))
        else:
            raise ValueError("maps must be I, [I], or [I,Q,U]")
    else:
        I = np.asarray(maps)
        Q = U = None

    bl = np.asarray(bl)

    npix = I.size
    nside_in = hp.npix2nside(npix)
    if nside_out is None:
        nside_out = nside_in

    # ---- choose safe lmax ----
    lmax_max = 3 * nside_in - 1
    lmax_bl = bl.size - 1
    if lmax is None:
        lmax_use = min(lmax_max, lmax_bl)
    else:
        lmax_use = min(int(lmax), lmax_max, lmax_bl)

    # build filters
    bl0 = bl[: lmax_use + 1].copy()

    has_pol = (Q is not None) and (U is not None) and (Q.size == I.size) and (U.size == I.size)
    if has_pol:
        bl2 = bl[: lmax_use + 1].copy()
        bl2[:2] = 0.0

    # ---- mask handling ----
    # Define validity from I (you can tighten this to include Q/U too)
    valid = (I != hp.UNSEEN) & np.isfinite(I)

    fill_I = np.nanmedian(I[valid]) if np.any(valid) else 0.0

    I_work = I.copy()
    I_work[~valid] = fill_I
    I_work[valid] -= fill_I  # remove offset on valid pixels only

    if has_pol:
        Q_work = Q.copy()
        U_work = U.copy()
        Q_work[~valid] = 0.0
        U_work[~valid] = 0.0

    # ---- harmonic transform, filter at native nside ----
    if has_pol:
        alm_I, alm_E, alm_B = hp.map2alm([I_work, Q_work, U_work], lmax=lmax_use, pol=True)
        alm_I = hp.almxfl(alm_I, bl0)
        alm_E = hp.almxfl(alm_E, bl2)
        alm_B = hp.almxfl(alm_B, bl2)

        dI, dQ, dU = hp.alm2map((alm_I, alm_E, alm_B), nside=nside_in, pol=True, lmax=lmax_use)

        # downgrade after filtering (your preferred method)
        if int(nside_out) != int(nside_in):
            dI, dQ, dU = hp.ud_grade([dI, dQ, dU], int(nside_out))

    else:
        alm_I = hp.map2alm(I_work, lmax=lmax_use)
        alm_I = hp.almxfl(alm_I, bl0)
        dI = hp.alm2map(alm_I, nside=nside_in, lmax=lmax_use)

        if int(nside_out) != int(nside_in):
            dI = hp.ud_grade(dI, int(nside_out))

    # ---- propagate mask and restore offset ----
    # downgrade mask: valid if any subpixel was valid
    valid_out = hp.ud_grade(valid.astype(np.float32), int(nside_out)) > 0.0

    if has_pol:
        dI[valid_out] += fill_I
        dI[~valid_out] = hp.UNSEEN
        dQ[~valid_out] = hp.UNSEEN
        dU[~valid_out] = hp.UNSEEN
        return dI, dQ, dU
    else:
        dI[valid_out] += fill_I
        dI[~valid_out] = hp.UNSEEN
        return dI



def gen_pixel_window(nside, lmax=None):
    """Approximate pixel window function"""

    # Okay, so we are approximating the high-ell values of the pixel
    # window function by transforming a circular top beam into l-space.
    # This is pretty close. 
    theta = np.linspace(0,np.pi,100000) 
    # top hat beam 
    beam = np.zeros_like(theta) 
    pixel_area = hp.nside2pixarea(nside, degrees=True)
    beam[theta<np.radians(3.6/np.pi*0.5*pixel_area**0.5)] = 1
    bl = hp.beam2bl(beam, theta, lmax=lmax) 
    return  bl/bl[0] 

def create_bl(beam, theta, lmax, normalise=False):
    """Wrapper for generating B_l with richard's code"""
    bl_raw_spin0 = hp.beam2bl(beam, theta, lmax=lmax)
    nl = bl_raw_spin0[0]
    bl_raw_spin2 = bl_raw_spin0.copy()
    bl_raw_spin2[0:2] = 0  # Set l=0 and l=1 to zero for spin-2 field
    if normalise:
        bl_raw_spin0 /= nl
        bl_raw_spin2 /= nl
    return bl_raw_spin0, bl_raw_spin2 

def fix_poles(maps, beams, nside_out, fwhm, lmax=None):
    """Smooth Q and U with a spin-0 field, interpolate over poles"""

    # Spin - 2 transform
    alms = hp.map2alm(maps, lmax=lmax)
    dalms = [hp.almxfl(alm, beam) for alm, beam in zip(alms,beams)]
    output_maps = hp.alm2map(dalms, nside=nside_out)

    # Spin-0 transform
    alm_Q = hp.map2alm(maps[1], lmax=lmax)
    alm_U = hp.map2alm(maps[2], lmax=lmax)
    dalm_Q = hp.almxfl(alm_Q, beams[0])
    dalm_U = hp.almxfl(alm_U, beams[0])
    Q_0 = hp.alm2map(dalm_Q, nside=nside_out)
    U_0 = hp.alm2map(dalm_U, nside=nside_out)

    # Now calculate theta, phi for all pixels 
    pixels = np.arange(hp.nside2npix(nside_out))
    theta, phi = hp.pix2ang(nside_out, pixels) 

    # We want a cosine window function in theta that starts at 2*fwhm and is fwhm/5 wide 
    wl = fwhm 
    window = (np.cos(2*np.pi*(theta - fwhm*2)/wl) + 1)*0.5
    window[theta < fwhm*2] = 1
    window[theta > fwhm*2 + wl/2] = 0

    # Now we want to fold together the two maps 
    output_maps[1] = output_maps[1]*(1-window) + Q_0*window
    output_maps[2] = output_maps[2]*(1-window) + U_0*window

    return output_maps

def _as_var_triplet(var_maps):
    """Return (II, QQ, UU, mode) where mode is 'I' or 'IQU'."""
    if isinstance(var_maps, (list, tuple)):
        if len(var_maps) == 1:
            return np.asarray(var_maps[0]), None, None, "I"
        if len(var_maps) == 3:
            II, QQ, UU = var_maps
            return np.asarray(II), np.asarray(QQ), np.asarray(UU), "IQU"
        if len(var_maps) == 6:
            II, QQ, UU = var_maps[:3]  # ignore cross terms
            return np.asarray(II), np.asarray(QQ), np.asarray(UU), "IQU"
        raise ValueError("var_maps must be II, [II], [II,QQ,UU], or [II,QQ,UU,IQ,IU,QU]")
    else:
        return np.asarray(var_maps), None, None, "I"


def smooth_variance_maps(var_maps, bl_spin0, lmax=None, nside_out=None, theta_samples=20000):
    """
    Smooth variance maps assuming uncorrelated per-pixel noise:
      Var_out(p) ≈ Ω_in ∫ W(θ)^2 Var_in(q) dΩ
    where W is the *signal* smoothing kernel whose harmonic coefficients are bl_spin0.

    Parameters
    ----------
    var_maps : II or [II] or [II,QQ,UU] or [II,QQ,UU,IQ,IU,QU]
        Variance maps. Cross terms ignored if present.
    bl_spin0 : array-like
        Harmonic kernel K_l applied to signal T (spin-0). (This is your transfer kernel.)
    lmax : int, optional
        Max multipole for filtering; clipped to min(3*nside_in-1, len(bl_spin0)-1).
    nside_out : int, optional
        Output nside. If None, keep input nside.
    theta_samples : int
        Sampling for constructing beam profile W(θ) and its squared harmonic transform.

    Returns
    -------
    If input was II only: II_out
    If input was triplet: [II_out, QQ_out, UU_out]
    """
    II, QQ, UU, mode = _as_var_triplet(var_maps)

    nside_in = hp.npix2nside(II.size)
    if nside_out is None:
        nside_out = nside_in
    nside_out = int(nside_out)


    bl_spin0 = np.asarray(bl_spin0) 
    lmax_max = 3 * nside_in - 1
    lmax_bl = bl_spin0.size - 1
    if lmax is None:
        lmax_use = min(lmax_max, lmax_bl)
    else:
        lmax_use = min(int(lmax), lmax_max, lmax_bl)

    # pixel window 
    w_pix_out = gen_pixel_window(nside_out, lmax=lmax_use)
    w_pix_in  = gen_pixel_window(nside_in , lmax=lmax_use)
    bl_spin0 = bl_spin0[: lmax_use + 1] * (w_pix_out/w_pix_in) 

    # Build real-space kernel W(θ) from the signal kernel bl_spin0,
    # then build variance kernel from W(θ)^2.
    theta = np.linspace(0.0, np.pi, theta_samples)
    W = hp.bl2beam(bl_spin0[: lmax_use + 1], theta)          # W(θ)
    W2 = W * W                                               # W(θ)^2
    bl_var = hp.beam2bl(W2, theta, lmax=lmax_use)            # harmonic coeffs of W^2

    # Pixel area factor (Ω_in) for per-pixel variances (see derivation)
    omega_in = float(hp.nside2pixarea(nside_in))

    def _filter_one(var_in: np.ndarray) -> np.ndarray:
        var_in = np.asarray(var_in)

        valid = (var_in != hp.UNSEEN) & np.isfinite(var_in)

        work = var_in.copy()

        alm = hp.map2alm(work, lmax=lmax_use)
        alm = hp.almxfl(alm, bl_var[: lmax_use + 1])
        sm = hp.alm2map(alm, nside=nside_in, lmax=lmax_use)

        # downgrade after filtering (your preferred robustness)
        if nside_out != nside_in:
            sm = hp.ud_grade(sm, nside_out)

        # propagate validity mask (valid if any subpixel was valid)
        valid_out = hp.ud_grade(valid.astype(np.float32), nside_out) > 0.5

        # convert integral-style smoothing to discrete per-pixel variance
        sm[valid_out] *= omega_in
        sm[~valid_out] = hp.UNSEEN

        return sm

    if mode == "I":
        return _filter_one(II)

    # IQU case
    out_II = _filter_one(II)
    out_QQ = _filter_one(QQ) if QQ is not None and QQ.size else np.empty(0)
    out_UU = _filter_one(UU) if UU is not None and UU.size else np.empty(0)
    return [out_II, out_QQ, out_UU]

def smooth_maps(
    m: Map,
    beam_info: BeamInfo,
    target_fwhm_arcmin: float = 0.0,
    nside_out: int = None,
) -> None:
    if nside_out is None:
        return
    if not np.isfinite(target_fwhm_arcmin) or target_fwhm_arcmin <= 0.0:
        return
    if beam_info is None or beam_info.beam_window is None:
        raise ValueError(f"{m.map_id}: no beam_info available; cannot smooth safely")

    if not np.isfinite(m.fwhm_arcmin) or m.fwhm_arcmin <= 0:
        raise ValueError(f"{m.map_id}: m.fwhm_arcmin not set; cannot decide smoothing direction")
    
    m.meta['native_fwhm_arcmin'] = m.fwhm_arcmin 


    lmax = int(beam_info.lmax)
    if target_fwhm_arcmin > m.fwhm_arcmin:
        gaussian_bl = hp.gauss_beam(np.radians(target_fwhm_arcmin/60.),lmax=lmax) 
        transfer_beam = np.zeros_like(gaussian_bl)
        gd = beam_info.beam_window != 0
        transfer_beam[gd] = gaussian_bl[gd]/np.maximum(beam_info.beam_window[gd], 1e-12)
    else: 
        transfer_beam = np.ones_like(beam_info.beam_window) 

    # --- map smoothing  ---
    v = valid_mask(m) 
    median_I = np.nanmedian(m.I[~v]) 
    m.I[~v] -= median_I
    if m.Q.size > 0:
        m.I, m.Q, m.U = smooth_map(
            [m.I, m.Q, m.U], transfer_beam, lmax=lmax, nside_out=nside_out
        )
    else:
        m.I = smooth_map(
            m.I, transfer_beam, lmax=lmax, nside_out=nside_out
        )
    v = valid_mask(m) 
    m.I[~v] += median_I

    # --- variance smoothing  ---
    if (m.II is not None) and (m.II.size > 0):
        if (m.Q.size > 0) and (m.QQ is not None) and (m.UU is not None) and (m.QQ.size == m.II.size) and (m.UU.size == m.II.size):
            # IQU variances available
            out_vars = smooth_variance_maps(
                [m.II, m.QQ, m.UU],
                bl_spin0=transfer_beam,
                lmax=lmax,
                nside_out=nside_out
            )
            m.II, m.QQ, m.UU = out_vars
        else:
            # Intensity-only variance 
            m.II = smooth_variance_maps(
                m.II,
                bl_spin0=transfer_beam,
                lmax=lmax,
                nside_out=nside_out
            )

    m.mask = (m.I < -1e20)
    m.fwhm_arcmin = target_fwhm_arcmin
    m.nside = nside_out 
    