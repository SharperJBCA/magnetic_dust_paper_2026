import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mde_pipeline.fitting.data_types import build_components_from_yaml
from mde_pipeline.utils.config import load_yaml



def _load_fitter_config(path: str, tmp_path: Path) -> dict:
    cfg = load_yaml(Path(path))["fitter"]
    cfg = copy.deepcopy(cfg)

    sims_dir = tmp_path / "sim_inputs"
    sims_dir.mkdir(parents=True, exist_ok=True)
    (sims_dir / "gains.json").write_text(json.dumps({}))
    cfg["sims_h5"] = str(sims_dir / "simulations.h5")
    return cfg



def test_build_components_accepts_scalar_init_and_derives_widths(tmp_path: Path):
    cfg = _load_fitter_config("configs/fitting/fitter.yaml", tmp_path)

    _, param0, widths0, _, _ = build_components_from_yaml(cfg)

    assert param0["A_s"] == 1.0
    # fallback for scalar init without normal prior sigma: 0.05 * abs(value)
    assert widths0["A_s"] == 0.05

    assert param0["beta_s"] == -3.1
    # from normal prior sigma
    assert widths0["beta_s"] == 0.5



def test_build_components_accepts_pair_init_list(tmp_path: Path):
    cfg = _load_fitter_config("configs/fitting/fitter_sytdffspmd.yaml", tmp_path)

    _, param0, widths0, _, _ = build_components_from_yaml(cfg)

    assert param0["beta_s"] == -3.1
    assert widths0["beta_s"] == 0.05
    assert param0["Te"] == 7000.0
    assert widths0["Te"] == 2000.0
