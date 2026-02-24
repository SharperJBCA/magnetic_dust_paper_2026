from pathlib import Path
import json
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mde_pipeline.io.maps_io import Map
from mde_pipeline.simulations import run


class DummyComponent:
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, nu_ghz, T, params, ctx):
        return {"I": np.ones(4, dtype=np.float32)}


class DummyTemplate:
    def __init__(self):
        self.name = "dummy"
        self.m = type("TemplateMap", (), {"nside": 1, "coord": "G", "I": np.ones(4, dtype=np.float32)})()


class DummyMapIO:
    writes = 0

    def __init__(self, *_args, **_kwargs):
        self.is_output = _args[1] == "simulations.h5"

    def read_map(self, target_name):
        m = Map(map_id=target_name, stage="processed")
        m.I = np.zeros(4, dtype=np.float32)
        m.II = np.ones(4, dtype=np.float32)
        m.mask = np.zeros(4, dtype=bool)
        m.nside = 1
        m.coord = "G"
        m.freq_ghz = 100.0
        m.calerr = 0.01
        return m

    def write_map(self, _m):
        DummyMapIO.writes += 1


def test_run_simulations_calls_gain_noise_and_write_once_per_target(tmp_path, monkeypatch):
    DummyMapIO.writes = 0

    processed_h5 = tmp_path / "processed.h5"
    out_h5 = tmp_path / "simulations.h5"

    cfg = {
        "simulations": {
            "processed_h5": str(processed_h5),
            "out_h5": str(out_h5),
            "templates": "unused.yaml",
            "targets": ["target_a"],
            "components": [
                {"class": "dummy", "name": "comp1", "template": "tpl"},
                {"class": "dummy", "name": "comp2", "template": "tpl"},
                {"class": "dummy", "name": "comp3", "template": "tpl"},
            ],
            "gain": {"enabled": True},
            "noise": {"enabled": True},
            "qc": {"enabled": False},
        }
    }

    gain_calls = {"count": 0}
    noise_calls = {"count": 0}

    def _fake_gain(_m):
        gain_calls["count"] += 1
        return 1.23

    def _fake_noise(_m):
        noise_calls["count"] += 1

    monkeypatch.setattr(run, "load_yaml", lambda _p: cfg)
    monkeypatch.setattr(run, "load_templates_config", lambda *_args, **_kwargs: {"tpl": DummyTemplate()})
    monkeypatch.setattr(run, "MapIO", DummyMapIO)
    monkeypatch.setitem(run.COMPONENTS, "dummy", DummyComponent)
    monkeypatch.setattr(run, "add_gain", _fake_gain)
    monkeypatch.setattr(run, "add_noise", _fake_noise)
    monkeypatch.setattr(run, "plot_spectrum", lambda *_args, **_kwargs: None)

    run.run_simulations(Path("dummy.yaml"), overwrite=False, dry_run=False)

    assert gain_calls["count"] == 1
    assert noise_calls["count"] == 1
    assert DummyMapIO.writes == 1

    gains_path = out_h5.parent / "gains.json"
    assert gains_path.exists()
    gains_data = json.loads(gains_path.read_text())
    assert gains_data == {"target_a": 1.23}
