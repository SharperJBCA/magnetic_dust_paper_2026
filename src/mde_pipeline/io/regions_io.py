from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Iterable, Tuple
import numpy as np
import h5py


@dataclass
class RegionSuite:
    region_map: np.ndarray              # int32 [npix]
    region_ids: np.ndarray              # int32 [nregions]
    region_names: List[str]             # list length nregions
    masks: Dict[str, np.ndarray]        # name -> bool [npix]
    attrs: Dict[str, object]            # suite-level attrs

    def get_mask(self, name: str) -> np.ndarray:
        if name not in self.masks:
            raise KeyError(f"Unknown region '{name}'. Available: {sorted(self.masks.keys())}")
        return self.masks[name]

    def get_pixels(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.get_mask(name))


class RegionsIO:
    def __init__(self, regions_filename: Path, tag: str):
        self.regions_filename = Path(regions_filename)
        self.tag = str(tag)

        self.region_map: Optional[np.ndarray] = None
        self.region_masks: Dict[str, np.ndarray] = {}
        self.region_ids: Optional[np.ndarray] = None
        self.region_names: Optional[List[str]] = None
        self.attrs: Dict[str, object] = {}

    # ---------- helpers ----------
    @staticmethod
    def _ensure_group(h: h5py.File, path: str) -> h5py.Group:
        return h.require_group(path)

    # ---------- public API ----------
    def list_suites(self) -> List[str]:
        if not self.regions_filename.exists():
            return []
        with h5py.File(self.regions_filename, "r") as h:
            if self.tag not in h:
                return []
            return list(h[self.tag].keys())

    def load_regions(self, region_group: str) -> RegionSuite:
        with h5py.File(self.regions_filename, "r") as h:
            grp = h[f"{self.tag}/{region_group}"]

            self.region_map = grp["region_map"][...].astype(np.int32)

            # Optional index table (recommended)
            if "region_ids" in grp and "region_names" in grp:
                self.region_ids = grp["region_ids"][...].astype(np.int32)
                # region_names may come back as bytes
                names = grp["region_names"][...]
                self.region_names = [n.decode("utf-8") if isinstance(n, (bytes, np.bytes_)) else str(n) for n in names]
            else:
                self.region_ids = None
                self.region_names = None

            self.region_masks = {}
            if "masks" in grp:
                for mask_name, dset in grp["masks"].items():
                    self.region_masks[str(mask_name)] = dset[...].astype(bool)

            # attrs
            self.attrs = dict(grp.attrs)

        return RegionSuite(
            region_map=self.region_map,
            region_ids=self.region_ids if self.region_ids is not None else np.array([], dtype=np.int32),
            region_names=self.region_names if self.region_names is not None else sorted(self.region_masks.keys()),
            masks=self.region_masks,
            attrs=self.attrs,
        )

    def write_regions(
        self,
        region_group: str,
        region_map: np.ndarray,
        masks: Dict[str, np.ndarray],
        attrs: Optional[Dict[str, object]] = None,
        overwrite: bool = True
    ) -> None:
        """
        Writes one region suite:
          /<tag>/<region_group>/region_map
          /<tag>/<region_group>/masks/<name>
          /<tag>/<region_group>/region_ids, region_names (recommended)
          attrs on the suite group
        """
        region_map = np.asarray(region_map, dtype=np.int32)

        # stable ordering for ids
        region_names = sorted(masks.keys())
        region_ids = np.arange(1, len(region_names) + 1, dtype=np.int32)

        with h5py.File(self.regions_filename, "a") as h:
            tag_grp = self._ensure_group(h, self.tag)

            if region_group in tag_grp:
                if overwrite:
                    del tag_grp[region_group]
                else:
                    raise FileExistsError(f"Region group '{self.tag}/{region_group}' already exists.")

            grp = tag_grp.create_group(region_group)

            # datasets
            grp.create_dataset(
                "region_map",
                data=region_map,
                shuffle=True,
            )

            grp.create_dataset("region_ids", data=region_ids)
            # store names as fixed/variable-length UTF-8 strings
            dt = h5py.string_dtype(encoding="utf-8")
            grp.create_dataset("region_names", data=np.array(region_names, dtype=dt))

            mask_grp = grp.create_group("masks")
            for name in region_names:
                mask = np.asarray(masks[name])
                if mask.shape != region_map.shape:
                    raise ValueError(f"Mask '{name}' has shape {mask.shape}, expected {region_map.shape}")

                mask_grp.create_dataset(
                    name,
                    data=mask,
                    shuffle=True,
                )

            # attrs
            if attrs:
                for k, v in attrs.items():
                    grp.attrs[k] = v

            # useful defaults
            grp.attrs.setdefault("npix", int(region_map.size))

    # ---------- convenience ----------
    def get_mask(self, name: str) -> np.ndarray:
        if name not in self.region_masks:
            raise KeyError(f"Unknown region '{name}'. Available: {sorted(self.region_masks.keys())}")
        return self.region_masks[name]

    def get_pixels(self, name: str) -> np.ndarray:
        return np.flatnonzero(self.get_mask(name))

    def get_region_id(self, name: str) -> int:
        if self.region_names is None or self.region_ids is None:
            # fall back to alphabetical order if no table
            names = sorted(self.region_masks.keys())
            return names.index(name) + 1
        return int(self.region_ids[self.region_names.index(name)])
