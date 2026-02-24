from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mde_pipeline.fisher.grid_workflow import _build_jobs_from_grid_section, _filter_component_lists
from mde_pipeline.fisher import grid_workflow


def test_build_jobs_from_grid_section_cartesian_product():
    cfg = {
        "grid": {
            "parameters": [
                {"name": "omega0_THz", "values": [0.5, 0.7]},
                {"name": "chi0", "values": [1.0, 1.3]},
            ]
        }
    }

    jobs = _build_jobs_from_grid_section(cfg)

    assert len(jobs) == 4
    assert jobs[0]["job_id"] == "j0000"
    assert "omega0_THz" in jobs[0]["params"]
    assert "chi0" in jobs[0]["params"]


def test_filter_component_lists_supports_thermaldust_alias():
    sim_cfg = {"simulations": {"components": [{"name": "dust"}, {"name": "synch"}]}}
    fit_cfg = {
        "fitter": {
            "components": [
                {"name": "dust", "params_map": {}},
                {"name": "synchrotron", "params_map": {}},
            ]
        }
    }

    _filter_component_lists(
        sim_cfg,
        fit_cfg,
        keep_sim_components=["dust"],
        keep_fitter_components=["thermaldust"],
    )

    assert [c["name"] for c in sim_cfg["simulations"]["components"]] == ["dust"]
    assert [c["name"] for c in fit_cfg["fitter"]["components"]] == ["dust"]


def test_filter_component_lists_supports_thermaldust_alias_for_sim_components():
    sim_cfg = {"simulations": {"components": [{"name": "dust"}, {"name": "synch"}]}}
    fit_cfg = {"fitter": {"components": [{"name": "dust", "params_map": {}}]}}

    _filter_component_lists(
        sim_cfg,
        fit_cfg,
        keep_sim_components=["thermaldust"],
        keep_fitter_components=["dust"],
    )

    assert [c["name"] for c in sim_cfg["simulations"]["components"]] == ["dust"]


def test_grid_workflow_overrides_fitter_out_dir(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"jobs": [{"job_id": "j0000", "run_name": "j0000", "params": {"x": 1.0, "y": 2.0}}]}))

    sims_cfg = tmp_path / "sims.yaml"
    sims_cfg.write_text("simulations:\n  components: []\n")

    fitter_cfg = tmp_path / "fitter.yaml"
    fitter_cfg.write_text(
        "fitter:\n"
        "  out_dir: products/fits\n"
        "  components: []\n"
    )

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("{}\n")
    templates_yaml = tmp_path / "templates.yaml"
    templates_yaml.write_text("{}\n")

    captured = {}

    monkeypatch.setattr(grid_workflow, "run_simulations", lambda *args, **kwargs: None)

    def _fake_run_fisher(*, fitter_yaml, **kwargs):
        payload = grid_workflow.load_yaml(fitter_yaml)
        captured["out_dir"] = payload["fitter"]["out_dir"]

    monkeypatch.setattr(grid_workflow, "run_fisher", _fake_run_fisher)
    monkeypatch.setattr(grid_workflow, "save_grid_outputs", lambda **kwargs: tmp_path / "ok")

    out_dir = tmp_path / "products" / "fits" / "v001"
    grid_workflow.run_fisher_grid_workflow(
        grid_manifest=manifest_path,
        sims_cfg=sims_cfg,
        fitter_cfg=fitter_cfg,
        data_yaml=data_yaml,
        templates_yaml=templates_yaml,
        regions_h5=tmp_path / "regions.h5",
        processed_h5=tmp_path / "processed.h5",
        out_dir=out_dir,
        grid_tag="grid_tag",
        x_param="x",
        y_param="y",
        region="highlat1",
        dataset_sets=["baseline"],
    )

    assert captured["out_dir"] == str(out_dir)
