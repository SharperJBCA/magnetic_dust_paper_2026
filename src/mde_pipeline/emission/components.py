from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Sequence, Optional
import numpy as np

COMPONENTS = {}

def register_component(cls):
    COMPONENTS[cls.__name__] = cls
    return cls

def _as_array(x, like: np.ndarray) -> np.ndarray:
    """Allow scalar or map parameter."""
    if np.isscalar(x):
        return np.full_like(like, float(x), dtype=like.dtype)
    x = np.asarray(x)
    if x.shape != like.shape:
        raise ValueError(f"Parameter map has shape {x.shape}, expected {like.shape}")
    return x



class EmissionComponent:
    stokes_out: Sequence[str] = ("I",)

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__

    def evaluate(self, nu_ghz: float, T, params: Dict[str, Any], ctx=None) -> Dict[str, np.ndarray]:
        raise NotImplementedError

@register_component
class SynchPowerLaw(EmissionComponent):
    stokes_out = ("I","Q","U")

    def evaluate(self, nu_ghz, T, params, ctx=None):
        A = params["A"]
        beta = params["beta"]
        nu0 = params["nu0_ghz"]

        s = self._powerlaw(nu_ghz, nu0, beta)

        I = _as_array(A, T.I) * T.I * s
        out = {"I": I}
        if getattr(T.m, "has_pol", False) and T.m.has_pol:
            out["Q"] = _as_array(A, T.Q) * T.Q * s
            out["U"] = _as_array(A, T.U) * T.U * s
        else:
            # If template has no Q/U but you asked for IQU, produce zeros (or raise).
            z = np.zeros_like(I)
            out["Q"], out["U"] = z, z
        return out
    
    def _powerlaw(self, nu, nu0, beta):
        return (nu / nu0) ** beta


@register_component
class ThermalDust(EmissionComponent):
    """
    Thermal dust as a modified blackbody:
        I(nu) ∝ A * (nu/nu0)^beta_d * B_nu(T_d)

    Implemented as a *shape ratio* relative to nu0, so your template T.I is treated as the
    dust map at nu0 in the pipeline's canonical unit system.
    """
    stokes_out = ("I", "Q", "U")

    def evaluate(self, nu_ghz, T, params, ctx=None):
        A = params["A"]
        beta_d = float(params["beta_d"])
        T_d = float(params["T_d"])
        nu0 = float(params["nu0_ghz"])

        nu_hz = float(nu_ghz) * 1e9
        nu0_hz = float(nu0) * 1e9

        # MBB scaling normalized at nu0
        pl = (nu_ghz / nu0) ** (beta_d -2.0) # convert to temperature
        bb = self._planck_Bnu(nu_hz, T_d) / self._planck_Bnu(nu0_hz, T_d)
        s = pl * bb

        I = _as_array(A, T.I) * T.I * s
        out = {"I": I}

        if T.m.has_pol:
            out["Q"] = _as_array(A, T.Q) * T.Q * s
            out["U"] = _as_array(A, T.U) * T.U * s
        else:
            z = np.zeros_like(I)
            out["Q"], out["U"] = z, z

        return out


    def _planck_Bnu(self, nu_hz: float, T: float) -> float:
        """
        Planck function B_nu(T) [W sr^-1 m^-2 Hz^-1] up to an overall constant.
        For our ratio B_nu/B_nu0, absolute units cancel, but we include full form for clarity.
        """
        # Physical constants (SI)
        _H = 6.62607015e-34   # Planck constant [J s]
        _KB = 1.380649e-23    # Boltzmann constant [J/K]
        _C = 299792458.0      # speed of light [m/s]

        if nu_hz <= 0:
            raise ValueError("nu_hz must be > 0")
        if T <= 0:
            raise ValueError("T must be > 0")

        x = (_H * nu_hz) / (_KB * T)
        # Use expm1 for numerical stability at small x
        denom = np.expm1(x)
        return (2.0 * _H * nu_hz**3 / _C**2) / denom

@register_component
class FreeFree(EmissionComponent):
    """
    Free-free (thermal bremsstrahlung) intensity component with spectral curvature
    from the velocity-averaged Gaunt factor.

    Model shape (optically thin):
        Tb(nu) ∝ Te^(-1/2) * nu^(-2) * g_ff(nu, Te)

    Notes:
      - This is a *shape* model; EM (or Hα→EM conversion) lives in the template T.I and/or A.
      - If you ever need optically thick turnover at low ν for high EM, we can extend this
        to use tau_ff and Tb = Te*(1-exp(-tau)).
    """
    stokes_out = ("I",)

    def evaluate(self, nu_ghz, T, params, ctx=None):
        Te = float(params["Te"])

        s = self.freefree_tb_shape(nu_ghz, Te) 
        I = T.I * s
        return {"I": I}

    @staticmethod
    def gaunt_factor(nu_ghz: float, Te: float) -> float:
        """
        Approximate velocity-averaged free-free Gaunt factor g_ff(nu, Te).
        This form provides the desired gentle curvature across GHz frequencies.
        """
        if nu_ghz <= 0:
            raise ValueError("nu_ghz must be > 0")
        if Te <= 0:
            raise ValueError("Te must be > 0")

        T4 = Te / 1.0e4
        # argument inside the log: nu_GHz * T4^{-3/2}
        x = nu_ghz * (T4 ** (-1.5))

        # Draine-style smooth approximation:
        a = 5.960 - (np.sqrt(3.0) / np.pi) * np.log(x)

        x = (np.log(4.995e-2 * nu_ghz**-1) + 1.5*np.log(Te))
        a = 0.366 * nu_ghz**0.1 * Te**-0.15 * x 
        return a

    @classmethod
    def freefree_tb_shape(cls, nu_ghz: float, Te: float) -> float:
        """
        Shape proportional to brightness temperature in the optically thin limit.
        Constants (EM, unit conversions) cancel in ratios.
        """
        gff = cls.gaunt_factor(nu_ghz, Te)
        T4 = Te/1e4
        return 8.396 * 1e-3 * nu_ghz**-2.1 * T4**0.667*10**(0.029/T4)*(1.0 + 0.08 ) * gff


@register_component
class SpinningDustLogNorm(EmissionComponent):
    stokes_out = ("I",)

    def evaluate(self, nu_ghz, T, params, ctx=None):
        A = np.exp(params["A"])
        W = params["W"]
        nu_peak = params["nu0_peak"]
        nu0 = params["nu0_ghz"]

        s = self.lognorm_shape(nu_ghz, nu_peak, W) / self.lognorm_shape(nu0, nu_peak, W)
        I = _as_array(A, T.I) * T.I * s
        return {"I": I}

    def lognorm_shape(self, nu, nu_peak, W):
        x = np.log(nu / nu_peak)
        return np.exp(-0.5 * (x * x) / W)



@register_component
class MagneticDustInclusions(EmissionComponent):
    stokes_out = ("I","Q","U")

    def sigmoid(self,x):
        return 1./(1 + np.exp(-x))

    def logit(self,x):
        return np.log(x/(1-x))

    def evaluate(self, nu_ghz, T, params, ctx=None):
        A = np.exp(params["A"])
        chi0 = np.exp(params["chi0"])
        phi = self.sigmoid(params["phi"])
        nu0 = params["nu0_ghz"]
        omega0_THz = np.exp(params["omega0_THz"])

        Td = params['Td']
        L_short = params['L_short']
        s,p = self.mde_inclusions_spectrum(nu_ghz, chi0, omega0_THz, phi, Td, L_short)
        s_norm = self.mde_inclusions_spectrum(nu0, chi0, omega0_THz, phi, Td, L_short)[0]
        s /= s_norm

        I = _as_array(A, T.I) * T.I * s
        out = {"I": I}

        if T.m.has_pol:
            out["Q"] = _as_array(A, T.Q) * T.Q * s * p 
            out["U"] = _as_array(A, T.U) * T.U * s * p
        else:
            z = np.zeros_like(I)
            out["Q"], out["U"] = z, z

        return out


    def mde_inclusions_spectrum(self, nu_ghz: float, chi0: float, omega0_THz: float, phi: float, Td: float, L_short: float) -> float:
        """
        """
        L_long = (1 - L_short)/2. 
        omega0 = omega0_THz*1e12 

        tau = 2.0/omega0 # critical 
        chi = self.chi(nu_ghz, chi0, omega0, tau) 

        chi_eff = self.chi_eff(chi, phi) 

        cabs_short_axis = self.cabs(nu_ghz, chi_eff, L_short) 
        cabs_long_axis = self.cabs(nu_ghz, chi_eff, L_long) 

        I = (1./3.) * (cabs_short_axis + cabs_long_axis) * self._planck_Bnu(nu_ghz*1e9, Td)
        p = (cabs_short_axis - cabs_long_axis)/(cabs_short_axis + cabs_long_axis)
        return I * (nu_ghz)**-2, p

    def cabs(self, nu_ghz, chi, L:float): 
        _C = 299792458.0      # speed of light [m/s]
        omega = 2*np.pi*nu_ghz*1e9 
        denom = np.abs(1 + L * chi)**2 
        return omega/_C * (np.imag(chi)/denom) 

    def chi(self, nu_ghz, chi0: float, omega_0: float, tau: float): 
        omega = 2*np.pi*nu_ghz*1e9 
        denom = 1 - (omega/omega_0)**2 - 1j*omega*tau 

        return chi0/denom 

    def chi_eff(self, chi, phi:float): 
        # TODO: add super paramagnetic part 
        if phi < 0:
            return 0.0
        numer = 2./3. * phi * chi 
        denom = 1 + (1./3) * chi *(1-2*phi/3.) 

        return numer/denom 
    
    def _planck_Bnu(self, nu_hz: float, T: float) -> float:
        """
        Planck function B_nu(T) [W sr^-1 m^-2 Hz^-1] up to an overall constant.
        For our ratio B_nu/B_nu0, absolute units cancel, but we include full form for clarity.
        """
        # Physical constants (SI)
        _H = 6.62607015e-34   # Planck constant [J s]
        _KB = 1.380649e-23    # Boltzmann constant [J/K]
        _C = 299792458.0      # speed of light [m/s]

        #if nu_hz <= 0:
        #    raise ValueError("nu_hz must be > 0")
        if T <= 0:
            raise ValueError("T must be > 0")

        x = (_H * nu_hz) / (_KB * T)
        denom = np.expm1(x)
        return (2.0 * _H * nu_hz**3 / _C**2) / denom
