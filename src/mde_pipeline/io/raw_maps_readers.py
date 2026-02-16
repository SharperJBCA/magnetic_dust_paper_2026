from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import healpy as hp 
from astropy.io import fits 

from ..io.maps_io import Map, BeamInfo

def dipole_map(nside, amplitude=0.0033621, phi=264.021, theta=48.253):
    direction = hp.ang2vec(phi, theta, lonlat=True)
    direction = direction/np.linalg.norm(direction)
    v = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))).T
    return amplitude * (v @ direction)

class BaseReader:
    def __init__(self, map_id:str, meta: dict):
        for name, value in meta.items(): 
            if name[0] == '_':
                continue 
            setattr(self,name, value) 
        self.map : Map = None
        self.beam_info : BeamInfo = None
        self.meta = meta
        self.map_id = map_id

        self.map = self.read() 
        self.beam_info = self.read_beam()

    def read(self) -> Map:
        raise NotImplementedError

    def read_beam(self) -> Optional[BeamInfo]:
        return None
    
    def get_stokes(self, stokes: str) -> Optional[np.ndarray]:
        if self.raw_map is None:
            return None
        return getattr(self.raw_map, stokes, None)

class LiteBIRDSimPTEPReader(BaseReader):

    def read(self): 
        II,QQ,UU = hp.read_map(self.filename, field=(0,1,2))

        I = np.zeros_like(II)
        Q = np.zeros_like(II)
        U = np.zeros_like(II)
        bad = (I < -1e20)  

        nside = hp.npix2nside(I.size)
        return Map(stage = 'raw',
                   map_id=self.map_id,
                    I=I, 
                    Q=Q, 
                    U=U, 
                    II=II, 
                    QQ=QQ, 
                    UU=UU, 
                    nside=nside,
                    mask=bad.astype(bool),
                    unit=getattr(self, "unit", None),
                    freq_ghz= getattr(self, "freq_ghz", None),
                    fwhm_arcmin=getattr(self, "fwhm_arcmin", None),
                    calerr=getattr(self,"calerr",None),
                    pol_convention=getattr(self,"pol_convention","COSMO"),
                    meta=self.meta)

    def read_beam(self):

            lmax = 3 * self.map.nside
            bl = hp.gauss_beam(np.radians(self.map.fwhm_arcmin/60.),lmax=lmax)
            return BeamInfo(
                lmax=len(bl) - 1,
                beam_window=np.asarray(bl),
                beam_type="Bl",
                meta={},
            )
class PlanckHFIPR3Map(BaseReader):

    def read(self): 
        I,Q,U,II,QQ,UU = hp.read_map(self.filename, field=(0,1,2,4,7,9))
        #I,Q,U,II,QQ,UU = np.zeros((6,12*64**2))
        bad = (I < -1e20)  
        I[bad] = hp.UNSEEN
        Q[bad] = hp.UNSEEN
        U[bad] = hp.UNSEEN
        II[bad] = hp.UNSEEN
        QQ[bad] = hp.UNSEEN
        UU[bad] = hp.UNSEEN
        nside = hp.npix2nside(I.size)
        return Map(stage = 'raw',
                   map_id=self.map_id,
                    I=I, 
                    Q=Q, 
                    U=U, 
                    II=II, 
                    QQ=QQ, 
                    UU=UU, 
                    nside=nside,
                    mask=bad.astype(bool),
                    unit=getattr(self, "unit", None),
                    freq_ghz= getattr(self, "freq_ghz", None),
                    fwhm_arcmin=getattr(self, "fwhm_arcmin", None),
                    calerr=getattr(self,"calerr",None),
                    pol_convention=getattr(self,"pol_convention","COSMO"),
                    meta=self.meta)

    def read_beam(self):
            if not hasattr(self, "beam_fits"):
                return None

            col = getattr(self, "beam_column", "TT_2_TT")
            with fits.open(self.beam_fits) as hdul:
                bl = hdul[1].data[col][0]

            return BeamInfo(
                lmax=len(bl) - 1,
                beam_window=np.asarray(bl),
                beam_type="Bl",
                meta={"beam_column": col, "beam_fits": self.beam_fits},
            )
    
class CosmoglobeMap(BaseReader):

    def read(self): 
        I,Q,U,II,QQ,UU = hp.read_map(self.filename, field=(0,1,2,3,4,5))
        bad = (I < -1e20)  
        I[bad] = hp.UNSEEN
        Q[bad] = hp.UNSEEN
        U[bad] = hp.UNSEEN
        II[bad] = hp.UNSEEN
        QQ[bad] = hp.UNSEEN
        UU[bad] = hp.UNSEEN

        nside = hp.npix2nside(I.size)
        I[~bad] -= dipole_map(nside)[~bad] * 1e6

        return Map(stage = 'raw',
                   map_id=self.map_id,
                    I=I, 
                    Q=Q, 
                    U=U, 
                    II=II, 
                    QQ=QQ, 
                    UU=UU, 
                    nside=nside,
                    mask=bad.astype(bool),
                    unit=getattr(self, "unit", None),
                    freq_ghz= getattr(self, "freq_ghz", None),
                    fwhm_arcmin=getattr(self, "fwhm_arcmin", None),
                    calerr=getattr(self,"calerr",None),
                    pol_convention=getattr(self,"pol_convention","COSMO"),
                    meta=self.meta)

    def read_beam(self):
            if not hasattr(self, "beam_fits"):
                return None

            with fits.open(self.beam_fits) as hdul:
                bl = hdul[self.beam_ext].data["BL"]

            return BeamInfo(
                lmax=len(bl) - 1,
                beam_window=np.asarray(bl),
                beam_type="Bl",
                meta={"beam_column": self.beam_ext, "beam_fits": self.beam_fits},
            )
    

class CosmoglobeCMBMap(BaseReader):

    def read(self): 
        I,Q,U = hp.read_map(self.filename, field=(0,1,2))
        bad = (I < -1e20)  
        I[bad] = hp.UNSEEN
        Q[bad] = hp.UNSEEN
        U[bad] = hp.UNSEEN

        nside = hp.npix2nside(I.size)
        return Map(stage = 'raw',
                   map_id=self.map_id,
                    I=I, 
                    Q=Q, 
                    U=U, 
                    nside=nside,
                    mask=bad.astype(bool),
                    unit=getattr(self, "unit", None),
                    freq_ghz= getattr(self, "freq_ghz", None),
                    fwhm_arcmin=getattr(self, "fwhm_arcmin", None),
                    calerr=getattr(self,"calerr",None),
                    pol_convention=getattr(self,"pol_convention","COSMO"),
                    meta=self.meta)

    def read_beam(self):
            if not hasattr(self, "beam_fits"):
                return None

            with fits.open(self.beam_fits) as hdul:
                bl = hdul[2].data["INT_BEAM"]

            return BeamInfo(
                lmax=len(bl) - 1,
                beam_window=np.asarray(bl),
                beam_type="Bl",
                meta={"beam_fits": self.beam_fits},
            )
    

    
class WHAMMap(BaseReader):

    def read(self): 
        I = hp.read_map(self.filename)
        bad = (I < -1e20)  
        I[bad] = hp.UNSEEN
        nside = hp.npix2nside(I.size)

        return Map(stage = 'raw',
                   map_id=self.map_id,
                    I=I, 
                    nside=nside,
                    mask=bad.astype(bool),
                    unit=getattr(self, "unit", None),
                    fwhm_arcmin=getattr(self,"fwhm_arcmin",None),
                    calerr=getattr(self,"calerr",None),
                    meta=self.meta)

    def read_beam(self):

            lmax = 3 * self.map.nside
            bl = hp.gauss_beam(np.radians(self.map.fwhm_arcmin/60.),lmax=lmax)
            return BeamInfo(
                lmax=len(bl) - 1,
                beam_window=np.asarray(bl),
                beam_type="Bl",
                meta={},
            )
    
class CBASSMap(BaseReader):

    def read(self): 
        I,Q,U,II,QQ,UU = hp.read_map(self.filename, field=(0,1,2,3,4,5))
        bad = (I < -1e20)  
        I[bad] = hp.UNSEEN
        Q[bad] = hp.UNSEEN
        U[bad] = hp.UNSEEN
        II[bad] = hp.UNSEEN
        QQ[bad] = hp.UNSEEN
        UU[bad] = hp.UNSEEN
        nside = hp.npix2nside(I.size)

        return Map(stage = 'raw',
                   map_id=self.map_id,
                    I=I, 
                    Q=Q, 
                    U=U, 
                    II=II, 
                    QQ=QQ, 
                    UU=UU, 
                    nside=nside,
                    mask=bad.astype(bool),
                    unit=getattr(self, "unit", None),
                    freq_ghz= getattr(self, "freq_ghz", None),
                    fwhm_arcmin=getattr(self, "fwhm_arcmin", None),
                    calerr=getattr(self,"calerr",None),
                    pol_convention=getattr(self,"pol_convention","IAU"),
                    meta=self.meta)

    def read_beam(self):

            lmax = 3 * self.map.nside
            bl = hp.gauss_beam(np.radians(self.map.fwhm_arcmin/60.),lmax=lmax)

            return BeamInfo(
                lmax=len(bl) - 1,
                beam_window=np.asarray(bl),
                beam_type="Bl",
                meta={},
            )
    
class CosmoglobeWMAPMap(BaseReader):

    def read(self): 
        I,Q,U,II,QQ,UU = hp.read_map(self.filename, field=(0,1,2,3,4,5))
        bad = (I < -1e20)  
        I[bad] = hp.UNSEEN
        Q[bad] = hp.UNSEEN
        U[bad] = hp.UNSEEN
        II[bad] = hp.UNSEEN
        QQ[bad] = hp.UNSEEN
        UU[bad] = hp.UNSEEN
        nside = hp.npix2nside(I.size)
        I[~bad] -= dipole_map(nside)[~bad] * 1e3

        return Map(stage = 'raw',
                   map_id=self.map_id,
                    I=I, 
                    Q=Q, 
                    U=U, 
                    II=II, 
                    QQ=QQ, 
                    UU=UU, 
                    nside=nside,
                    mask=bad.astype(bool),
                    unit=getattr(self, "unit", None),
                    freq_ghz= getattr(self, "freq_ghz", None),
                    fwhm_arcmin=getattr(self, "fwhm_arcmin", None),
                    calerr=getattr(self,"calerr",None),
                    pol_convention=getattr(self,"pol_convention","COSMO"),
                    meta=self.meta)

    def read_beam(self):

            lmax = 3 * self.map.nside
            bl = hp.gauss_beam(np.radians(self.map.fwhm_arcmin/60.),lmax=lmax)

            return BeamInfo(
                lmax=len(bl) - 1,
                beam_window=np.asarray(bl),
                beam_type="Bl",
                meta={},
            )
    
class SPASSMap(BaseReader):

    def read(self): 
        I = hp.read_map(self.filename[0]).astype(np.float64)
        Q = hp.read_map(self.filename[1]).astype(np.float64)
        U = hp.read_map(self.filename[2]).astype(np.float64)
        II = hp.read_map(self.filename[3]).astype(np.float64)**2
        bad = (np.abs(I) > 1e20) 
        I[bad] = hp.UNSEEN
        Q[bad] = hp.UNSEEN 
        U[bad] = hp.UNSEEN
        II[bad] = hp.UNSEEN 
        QQ = II.copy() 
        UU = II.copy()
        nside = hp.npix2nside(I.size)

        return Map(stage = 'raw',
                   map_id=self.map_id,
                    I=I, 
                    Q=Q, 
                    U=U, 
                    II=II, 
                    QQ=QQ, 
                    UU=UU, 
                    nside=nside,
                    mask=bad.astype(bool),
                    unit=getattr(self, "unit", None),
                    freq_ghz= getattr(self, "freq_ghz", None),
                    fwhm_arcmin=getattr(self, "fwhm_arcmin", None),
                    calerr=getattr(self,"calerr",None),
                    pol_convention=getattr(self,"pol_convention","IAU"),
                    meta=self.meta)

    def read_beam(self):

            lmax = 3 * self.map.nside
            bl = hp.gauss_beam(np.radians(self.map.fwhm_arcmin/60.),lmax=lmax)

            return BeamInfo(
                lmax=len(bl) - 1,
                beam_window=np.asarray(bl),
                beam_type="Bl",
                meta={},
            )
    
