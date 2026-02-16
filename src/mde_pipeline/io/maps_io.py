from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Literal

import copy 
import numpy as np
import h5py
import healpy as hp


Stage = Literal["raw", "processed", "sim"]

def _decode_h5_value(x: Any) -> Any:
    """Decode common h5py scalar/bytes types into plain Python types."""
    # bytes -> str
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")

    # numpy scalar -> python scalar
    if isinstance(x, np.generic):
        return x.item()

    # list/tuple of bytes -> list of str
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            if len(x) > 0 and isinstance(x[0], (bytes, np.bytes_)):
                return [y.decode("utf-8") for y in x]
        except Exception:
            pass

    return x


@dataclass
class BeamInfo:
    lmax: int
    beam_window: np.ndarray
    beam_type: str  # "Bl" or "theta"
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Map:
    map_id: str = ""
    stage: Stage = "raw"

    # Stokes maps
    I: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    Q: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    U: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    # Variances / weights (optional)
    II: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    QQ: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    UU: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    mask: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))

    # Core metadata (may be unset when raw)
    unit: str = ""
    fwhm_arcmin: float = float("nan")
    nside: Optional[int] = None
    coord: str = "G"
    freq_ghz: float = float("nan")
    calerr: float = float("nan")
    pol_convention: str = ""  # set explicitly; canonical could be "IAU"

    # Optional beam metadata (native or derived)
    beam: Optional[BeamInfo] = None

    # Anything else: provenance, filepaths, native conventions, notes, etc.
    meta: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience ----
    @property
    def data_fields(self):
        return ['I','Q','U','II','QQ','UU','mask']
    
    @property
    def attribute_fields(self):
        return ['unit','fwhm_arcmin','nside','coord','freq_ghz','calerr','pol_convention']

    @property 
    def stokes_list(self): 
        all_stokes = ['I','Q','U','II','QQ','UU']
        return [k for k in all_stokes if getattr(self,k).size > 0]

    @property
    def has_pol(self) -> bool:
        return self.Q.size > 0 and self.U.size > 0

    @property
    def has_variance(self) -> bool:
        return self.II.size > 0 and (self.QQ.size > 0 or not self.has_pol)

    @staticmethod
    def slice_map(m: Map, pix: np.ndarray) -> Map:
        m2 = Map(map_id=m.map_id, stage=m.stage)
        # copy attrs
        for k in m.attribute_fields:
            setattr(m2, k, getattr(m, k))
        m2.meta = dict(m.meta)
        # slice arrays
        for f in ["I","Q","U","II","QQ","UU","mask"]:
            arr = getattr(m, f)
            if arr.size:
                setattr(m2, f, arr[pix])
        return m2

    def stokes(self, stokes, pixels=None) -> np.ndarray:
        if isinstance(pixels,type(None)):
            pixels = np.arange(self.I,dtype='int')
        m = getattr(self, stokes)
        return m[pixels]

    def stokes_flat(self, pixels=None) -> np.ndarray:
        if isinstance(pixels,type(None)):
            pixels = np.arange(self.I,dtype='int')
        # Always return IQU-length vector for downstream matrix code
        if not self.has_pol:
            z = np.zeros_like(self.I[pixels])
            return np.hstack([self.I[pixels], z, z])
        if not (self.I.size == self.Q.size == self.U.size):
            raise ValueError(f"{self.map_id}: I/Q/U sizes mismatch: {self.I.size}/{self.Q.size}/{self.U.size}")
        return np.hstack([self.I[pixels], self.Q[pixels], self.U[pixels]])

    def variance_flat(self, pixels=None) -> np.ndarray:
        if isinstance(pixels,type(None)):
            pixels = np.arange(self.I,dtype='int')
        if self.II.size == 0:
            z = np.zeros_like(self.I[pixels])
            return np.hstack([z, z, z])
        if not self.has_pol:
            z = np.zeros_like(self.II[pixels])
            return np.hstack([self.II[pixels], z, z])
        if not (self.II.size == self.QQ.size == self.UU.size):
            raise ValueError(f"{self.map_id}: II/QQ/UU sizes mismatch: {self.II.size}/{self.QQ.size}/{self.UU.size}")
        return np.hstack([self.II[pixels], self.QQ[pixels], self.UU[pixels]])

    def ud_grade(
        self,
        nside_out: int,
        mask_threshold: float = 0.99,
        order_in: str = "RING",
        order_out: str = "RING",
        dtype: np.dtype | None = None,
    ) -> Map:
        """
        Change map resolution to `nside_out` using healpy.ud_grade.

        - I/Q/U are treated as "temperature-like" -> power=0
        - II/QQ/UU are treated as variances -> power=2 (scales with pixel area)
        - mask is treated as a coverage fraction then thresholded back to {0,1}

        Parameters
        ----------
        nside_out : int
            Target NSIDE.
        mask_threshold : float
            When degrading a boolean mask, keep a pixel if its averaged coverage
            fraction >= this threshold.
        order_in, order_out : str
            Healpix ordering ("RING" or "NESTED").
        dtype : np.dtype | None
            If provided, cast output arrays to this dtype (except mask -> uint8).

        Returns
        -------
        self : Map
            Modified in place.
        """
        if self.nside is None:
            raise ValueError(f"{self.map_id}: cannot ud_grade because nside is None")
        if int(nside_out) == int(self.nside):
            return self

        nside_out = int(nside_out)

        def _grade(arr: np.ndarray, power: int) -> np.ndarray:
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                return arr
            out = hp.ud_grade(
                arr,
                nside_out=nside_out,
                order_in=order_in,
                order_out=order_out,
                power=power,
            )
            if dtype is not None:
                out = out.astype(dtype, copy=False)
            return out

        # Stokes (temperature-like)
        for k in ("I", "Q", "U"):
            setattr(self, k, _grade(getattr(self, k), power=0))

        # Variances (area scaling)
        for k in ("II", "QQ", "UU"):
            setattr(self, k, _grade(getattr(self, k), power=2))

        # Mask: degrade as float "coverage fraction" then threshold back to 0/1
        if isinstance(self.mask, np.ndarray) and self.mask.size > 0:
            frac = hp.ud_grade(
                self.mask.astype(np.float32),
                nside_out=nside_out,
                order_in=order_in,
                order_out=order_out,
                power=0,
            )
            self.mask = (frac >= float(mask_threshold)).astype(bool)

        self.nside = nside_out
        return self

    @staticmethod
    def zeros_like(src: Map, *, map_id: Optional[str] = None, stage: Optional[Stage] = None) -> Map:
        """
        Create a new Map with metadata copied from `src`, but data arrays set to zeros
        and mask set to all False (0).

        Arrays are only created if present in `src` (i.e. size > 0), and will match
        shape + dtype.
        """
        def zlike(a: np.ndarray) -> np.ndarray:
            return np.zeros_like(a) if (isinstance(a, np.ndarray) and a.size > 0) else np.empty(0, dtype=np.float32)

        # Data arrays
        I  = zlike(src.I)
        Q  = zlike(src.Q)
        U  = zlike(src.U)
        II = zlike(src.II)
        QQ = zlike(src.QQ)
        UU = zlike(src.UU)

        # Mask: same length as I if possible, else match src.mask if that exists, else empty.
        if isinstance(src.I, np.ndarray) and src.I.size > 0:
            mask = np.zeros(src.I.shape, dtype=bool)  # all False
        elif isinstance(src.mask, np.ndarray) and src.mask.size > 0:
            mask = np.zeros(src.mask.shape, dtype=bool)
        else:
            mask = np.empty(0, dtype=bool)

        # Copy beam + meta safely (avoid shared references)
        beam_copy = copy.deepcopy(src.beam) if src.beam is not None else None
        meta_copy = copy.deepcopy(src.meta) if src.meta is not None else {}

        out = Map(
            map_id=(map_id if map_id is not None else src.map_id),
            stage=(stage if stage is not None else src.stage),

            I=I, Q=Q, U=U,
            II=II, QQ=QQ, UU=UU,
            mask=mask,

            unit=src.unit,
            fwhm_arcmin=src.fwhm_arcmin,
            nside=src.nside,
            coord=src.coord,
            freq_ghz=src.freq_ghz,
            calerr=src.calerr,
            pol_convention=src.pol_convention,

            beam=beam_copy,
            meta=meta_copy,
        )
        return out
    
class MapIO:
    def __init__(self, data_path: str = "products/processed_maps", 
                 filename: str = "processed_maps.h5"):
        self.data_path = Path(data_path)
        self.filename = filename

    def _file(self) -> Path:
        return self.data_path / self.filename

    def read_map(self, map_id: str) -> Map:            
        filepath = self._file()
        if not filepath.exists():
            raise FileNotFoundError(str(filepath))

        m = Map(map_id=map_id)

        with h5py.File(filepath, "r") as h:
            if map_id not in h:
                raise KeyError(f"{map_id} not in {filepath}")

            grp = h[map_id]

            # Load datasets only
            for name, obj in grp.items():
                if isinstance(obj, h5py.Dataset):
                    setattr(m, name, obj[...])

            # Load attrs (core fields + spillover meta)
            for name, val in grp.attrs.items():
                val = _decode_h5_value(val)
                if hasattr(m, name):
                    setattr(m, name, val)
                else:
                    m.meta[name] = val

        # Fix mask if it hasn't been changed - will fix at source soon
        m.mask = (m.I < -1e20)
        if m.II.size > 0:
            m.mask |= (m.II < -1e20)
        m.nside = hp.npix2nside(m.I.size)
        return m

    def write_map(
        self,
        map_data: Map,
    ) -> None:
        filepath = self._file()
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, "a") as h:
            if map_data.map_id in h:
                del h[map_data.map_id]
            grp = h.create_group(map_data.map_id)

            # datasets
            for field_name in map_data.data_fields:
                arr = getattr(map_data, field_name)
                if isinstance(arr, np.ndarray) and arr.size > 0:
                    grp.create_dataset(
                        field_name,
                        data=arr,
                        shuffle=True,
                        chunks=True,
                    )

            # attrs (core)
            for attr_name in map_data.attribute_fields:
                v = getattr(map_data, attr_name)
                if type(v) in [float,int,str]:
                    grp.attrs[attr_name] =v

            # attrs (extra)
            for k, v in map_data.meta.items():
                if type(v) in [float,int,str]:
                    grp.attrs[k] =v
