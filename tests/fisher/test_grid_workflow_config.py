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


def test_grid_workflow_reuses_simulation_h5_and_skips_simulations(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "jobs": [
                    {"job_id": "j0000", "run_name": "j0000", "params": {"x": 1.0, "y": 2.0}},
                    {"job_id": "j0001", "run_name": "j0001", "params": {"x": 1.2, "y": 2.2}},
                ]
            }
        )
    )

    sims_cfg = tmp_path / "sims.yaml"
    sims_cfg.write_text("simulations:\n  components: []\n")

    fitter_cfg = tmp_path / "fitter.yaml"
    fitter_cfg.write_text("fitter:\n  out_dir: products/fits\n  components: []\n")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("{}\n")
    templates_yaml = tmp_path / "templates.yaml"
    templates_yaml.write_text("{}\n")

    shared_sims = tmp_path / "shared_simulations.h5"
    run_simulations_calls = []
    captured_sims_h5 = []

    def _fake_run_simulations(*args, **kwargs):
        run_simulations_calls.append((args, kwargs))

    def _fake_run_fisher(*, fitter_yaml, **kwargs):
        payload = grid_workflow.load_yaml(fitter_yaml)
        captured_sims_h5.append(payload["fitter"]["sims_h5"])

    monkeypatch.setattr(grid_workflow, "run_simulations", _fake_run_simulations)
    monkeypatch.setattr(grid_workflow, "run_fisher", _fake_run_fisher)
    monkeypatch.setattr(grid_workflow, "save_grid_outputs", lambda **kwargs: tmp_path / "ok")

    grid_workflow.run_fisher_grid_workflow(
        grid_manifest=manifest_path,
        sims_cfg=sims_cfg,
        fitter_cfg=fitter_cfg,
        data_yaml=data_yaml,
        templates_yaml=templates_yaml,
        regions_h5=tmp_path / "regions.h5",
        processed_h5=tmp_path / "processed.h5",
        out_dir=tmp_path / "products" / "fits" / "v001",
        grid_tag="grid_tag",
        x_param="x",
        y_param="y",
        region="highlat1",
        dataset_sets=["baseline"],
        reuse_simulation_h5=shared_sims,
        skip_simulations=True,
    )

    assert run_simulations_calls == []
    assert captured_sims_h5 == [str(shared_sims), str(shared_sims)]


def test_grid_workflow_applies_gain_error_group_overrides_to_temp_fit_config(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"jobs": [{"job_id": "j0000", "run_name": "j0000", "params": {"x": 1.0, "y": 2.0}}]}))

    sims_cfg = tmp_path / "sims.yaml"
    sims_cfg.write_text("simulations:\n  components: []\n")

    fitter_cfg = tmp_path / "fitter.yaml"
    fitter_cfg.write_text(
        "fitter:\n"
        "  out_dir: products/fits\n"
        "  components: []\n"
        "  gains:\n"
        "    - name: planck100_pr3\n"
        "      param: cal_planck100_pr3\n"
        "      priors:\n"
        "        - type: normal\n"
        "          params:\n"
        "            cal_planck100_pr3: [1.0, 0.01]\n"
        "    - name: wmapK_cosmo\n"
        "      param: cal_wmapK_cosmo\n"
        "      priors:\n"
        "        - type: normal\n"
        "          params:\n"
        "            cal_wmapK_cosmo: [1.0, 0.02]\n"
        "    - name: haslam\n"
        "      param: cal_haslam\n"
        "      priors:\n"
        "        - type: normal\n"
        "          params:\n"
        "            cal_haslam: [1.0, 0.03]\n"
    )

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("{}\n")
    templates_yaml = tmp_path / "templates.yaml"
    templates_yaml.write_text("{}\n")

    monkeypatch.setattr(grid_workflow, "run_simulations", lambda *args, **kwargs: None)
    monkeypatch.setattr(grid_workflow, "run_fisher", lambda **kwargs: None)
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
        gain_error_groups={"planck": 0.05, "wmap": 0.07},
    )

    fit_tmp = out_dir / "grid_tag" / "_grid_tmp_configs" / "j0000_fit.yaml"
    fit_cfg_obj = grid_workflow.load_yaml(fit_tmp)
    gains = {g["name"]: g for g in fit_cfg_obj["fitter"]["gains"]}

    assert gains["planck100_pr3"]["priors"][0]["params"]["cal_planck100_pr3"][1] == 0.05
    assert gains["wmapK_cosmo"]["priors"][0]["params"]["cal_wmapK_cosmo"][1] == 0.07
    assert gains["haslam"]["priors"][0]["params"]["cal_haslam"][1] == 0.03


def test_run_fisher_grid_from_yaml_merges_model_and_grid_gain_error_groups(tmp_path, monkeypatch):
    grid_yaml = tmp_path / "grid.yaml"
    manifest_out = tmp_path / "runtime_manifest.json"
    grid_yaml.write_text(
        "workflow:\n"
        f"  manifest_out: {manifest_out}\n"
        "  sims_config: sims.yaml\n"
        "  fisher_config: fitter.yaml\n"
        "  data_config: data.yaml\n"
        "  templates_config: templates.yaml\n"
        "grid:\n"
        "  grid_tag: tag\n"
        "  x_param: x\n"
        "  y_param: y\n"
        "  region: r1\n"
        "  parameters:\n"
        "    - name: x\n"
        "      values: [1.0]\n"
        "    - name: y\n"
        "      values: [2.0]\n"
        "  gain_error_groups:\n"
        "    planck: 0.02\n"
        "model:\n"
        "  gain_error_groups:\n"
        "    planck: 0.01\n"
        "    wmap: 0.03\n"
    )

    captured = {}

    def _fake_run_fisher_grid_workflow(**kwargs):
        captured["gain_error_groups"] = kwargs["gain_error_groups"]
        return tmp_path / "ok"

    monkeypatch.setattr(grid_workflow, "run_fisher_grid_workflow", _fake_run_fisher_grid_workflow)

    grid_workflow.run_fisher_grid_from_yaml(
        grid_yaml=grid_yaml,
        tag="t1",
    )

    assert captured["gain_error_groups"] == {"planck": 0.02, "wmap": 0.03}


def test_grid_workflow_writes_mode_comparison_artifacts(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"jobs": [{"job_id": "j0000", "run_name": "j0000", "params": {"x": 1.0, "y": 2.0}}]})
    )

    sims_cfg = tmp_path / "sims.yaml"
    sims_cfg.write_text("simulations:\n  components: []\n")

    fitter_cfg = tmp_path / "fitter.yaml"
    fitter_cfg.write_text("fitter:\n  out_dir: products/fits\n  components: []\n")

    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("{}\n")
    templates_yaml = tmp_path / "templates.yaml"
    templates_yaml.write_text("{}\n")

    monkeypatch.setattr(grid_workflow, "run_simulations", lambda *args, **kwargs: None)

    def _fake_run_fisher(*, fitter_yaml, out_dir, run_name, region_ids, **kwargs):
        _ = grid_workflow.load_yaml(fitter_yaml)
        for set_name, sigma in (("baseline", 0.2), ("baseline_plus_litebird", 0.1)):
            summary_path = (
                out_dir / "grid_tag" / run_name / "fisher" / set_name / region_ids[0] / "summary.json"
            )
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps({"fiducial": {"A_md": 0.4}, "sigma_1d": {"A_md": sigma}}))

    monkeypatch.setattr(grid_workflow, "run_fisher", _fake_run_fisher)
    monkeypatch.setattr(grid_workflow, "save_grid_outputs", lambda **kwargs: tmp_path / "grid_maps")

    out_dir = tmp_path / "products" / "fits" / "v001"
    result = grid_workflow.run_fisher_grid_workflow(
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
        dataset_sets=["baseline", "baseline_plus_litebird"],
        stokes_modes=[["I", "Q", "U"], ["Q", "U"]],
    )

    assert result == out_dir / "grid_tag" / "fisher" / "grid_maps"
    comparison_path = out_dir / "grid_tag" / "fisher" / "snr_mode_comparison.json"
    assert comparison_path.exists()

    payload = json.loads(comparison_path.read_text())
    assert payload["region"] == "highlat1"
    assert "IQU" in payload["modes"]
    assert "QU" in payload["modes"]

    iqu = payload["modes"]["IQU"]["j0000"]
    assert iqu["snr_baseline"] == 2.0
    assert iqu["snr_plus"] == 4.0
    assert iqu["snr_abs_diff"] == 2.0
    assert iqu["snr_ratio"] == 2.0

    csv_path = out_dir / "grid_tag" / "fisher" / "grid_maps" / "snr_mode_comparison_table.csv"
    assert csv_path.exists()
