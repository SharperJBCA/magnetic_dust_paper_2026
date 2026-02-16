from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional

from ..io.maps_io import Map, MapIO
from ..utils.config import load_yaml

class TemplateBase:
    """
    Read-only wrapper around a Map that:
      - exposes I/Q/U/II/QQ/UU
      - provides filled versions (mask -> 0 by default)
      - caches derived products
    """

    def __init__(
        self,
        m: Map,
        name: str = "",
        fill_value: float = 0.0,
    ):
        self.m = m
        self.name = name or m.map_id
        self.fill_value = float(fill_value)
        self._cache: Dict[str, np.ndarray] = {}

    # --- mask helpers ---
    @property
    def mask(self) -> np.ndarray:
        return self.m.mask

    @property
    def bad(self) -> np.ndarray:
        if self.m.mask.size == 0:
            return np.zeros(self.m.I.size, dtype=bool)
        return self.m.mask

    def _filled(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x
        if self.m.mask.size == 0:
            return x
        y = x.copy()
        y[self.bad] = self.fill_value
        return y
    
    @staticmethod
    def slice_template(T: TemplateBase, pix: np.ndarray) -> TemplateBase:
        # preserve subclass
        T2 = T.__class__.__new__(T.__class__)
        T2.__dict__ = dict(T.__dict__)          # shallow copy
        T2.m = Map.slice_map(T.m, pix)
        T2._cache = {}                          # reset cache!
        return T2

    # --- raw fields (exactly what's in Map) ---
    @property
    def I_raw(self) -> np.ndarray: return self.m.I
    @property
    def Q_raw(self) -> np.ndarray: return self.m.Q
    @property
    def U_raw(self) -> np.ndarray: return self.m.U
    @property
    def II_raw(self) -> np.ndarray: return self.m.II
    @property
    def QQ_raw(self) -> np.ndarray: return self.m.QQ
    @property
    def UU_raw(self) -> np.ndarray: return self.m.UU

    # --- filled fields (mask->fill_value) ---
    @property
    def I(self) -> np.ndarray: 
        return self._filled(self.m.I)

    @property
    def Q(self) -> np.ndarray: 
        return self._filled(self.m.Q)

    @property
    def U(self) -> np.ndarray: 
        return self._filled(self.m.U)

    @property
    def II(self) -> np.ndarray: 
        return self._filled(self.m.II)

    @property
    def QQ(self) -> np.ndarray: 
        return self._filled(self.m.QQ)
    
    @property
    def UU(self) -> np.ndarray: 
        return self._filled(self.m.UU)

    @property
    def IQU(self) -> np.ndarray:
        # returns (3, npix) filled
        if not self.m.has_pol:
            z = np.zeros_like(self.I)
            return np.vstack([self.I, z, z])
        return np.vstack([self.I, self.Q, self.U])

class TemplateDerivePol(TemplateBase):
    def __init__(
        self,
        m: Map,
        name: str = "",
        **kwargs,
    ):
        super().__init__(m, name=name or (m.map_id + ":dust_from_pol"), **kwargs)

    @property
    def P(self) -> np.ndarray:
        if "P" not in self._cache:
            self._cache["P"] = np.sqrt(self.Q**2 + self.U**2)
        return self._cache["P"]

    @property
    def p(self) -> np.ndarray:
        if "p" not in self._cache:
            I = self.I
            P = self.P
            p = np.zeros_like(I)
            good = (I != 0.0) & (~self.bad)
            p[good] = P[good] / I[good]
            self._cache["p"] = p
        return self._cache["p"]

    @property
    def psi(self) -> np.ndarray:
        if "psi" not in self._cache:
            psi = 0.5 * np.arctan2(self.U, self.Q)
            self._cache["psi"] = psi
        return self._cache["psi"]

    @property
    def Q(self) -> np.ndarray:
        if "Q_synth" not in self._cache:
            self._cache["Q_synth"] = self.p * self.I * np.cos(2.0 * self.psi)
        return self._cache["Q_synth"]

    @property
    def U(self) -> np.ndarray:
        if "U_synth" not in self._cache:
            self._cache["U_synth"] = self.p * self.I * np.sin(2.0 * self.psi)
        return self._cache["U_synth"]



TEMPLATES = {
    "TemplateBase": TemplateBase,
    "TemplateDerivePol": TemplateDerivePol,  
}

def load_templates_config(filename, out_h5: Path) -> Dict[str, TemplateBase]:
    """
    filename: YAML path, e.g.
      dust:
        template: planck353_pr3
        template_class: TemplateDerivePol
        template_kwargs: {fill_value: 0.0}
    """
    if isinstance(filename,dict):
        cfg = filename 
    else:
        cfg = load_yaml(filename)

    map_io = MapIO(data_path=str(out_h5.parent), filename=out_h5.name)

    templates: Dict[str, TemplateBase] = {}
    for name, info in cfg.items():
        cls_name = info.get("template_class", "TemplateBase")
        cls = TEMPLATES.get(cls_name)
        if cls is None:
            raise KeyError(f"Unknown template_class '{cls_name}' for '{name}'. "
                           f"Known: {sorted(TEMPLATES.keys())}")

        map_id = info["template"] 
        m = map_io.read_map(map_id)

        kwargs = dict(info.get("template_kwargs", {}))

        templates[name] = cls(m, name=name, **kwargs) 

    return templates
