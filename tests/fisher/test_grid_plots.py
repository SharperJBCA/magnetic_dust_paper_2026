from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mde_pipeline.fisher.grid_plots import build_snr_grid_table, save_grid_outputs


def _write_summary(path: Path, a_md: float, sigma: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sigma_1d": {"A_md": sigma},
        "fiducial": {"A_md": a_md},
    }
    path.write_text(json.dumps(payload))


def test_build_snr_grid_table_autodiscovers_j_runs(tmp_path):
    run1 = tmp_path / "j0000_omega0THz0.9_chi01.0_Amd0.2"
    run2 = tmp_path / "j0001_omega0THz1.1_chi01.5_Amd0.2"

    _write_summary(run1 / "fisher" / "baseline" / "north" / "summary.json", a_md=0.2, sigma=0.1)
    _write_summary(run2 / "fisher" / "baseline" / "north" / "summary.json", a_md=0.2, sigma=0.05)

    table = build_snr_grid_table(
        runs_root=tmp_path,
        x_param="omega0THz",
        y_param="chi0",
        region="north",
        dataset_sets=["baseline"],
    )

    assert len(table) == 2
    snr_values = sorted(r["snr"] for r in table)
    assert snr_values == [2.0, 4.0]


def test_save_grid_outputs_writes_npz_csv_and_plot(tmp_path):
    run1 = tmp_path / "j0000_omega0THz0.9_chi01.0_Amd0.2"
    run2 = tmp_path / "j0001_omega0THz1.1_chi01.0_Amd0.2"

    _write_summary(run1 / "fisher" / "baseline" / "north" / "summary.json", a_md=0.2, sigma=0.1)
    _write_summary(run2 / "fisher" / "baseline" / "north" / "summary.json", a_md=0.2, sigma=0.05)
    _write_summary(run1 / "fisher" / "baseline_plus_litebird" / "north" / "summary.json", a_md=0.2, sigma=0.05)
    _write_summary(run2 / "fisher" / "baseline_plus_litebird" / "north" / "summary.json", a_md=0.2, sigma=0.025)

    out_dir = save_grid_outputs(
        runs_root=tmp_path,
        x_param="omega0THz",
        y_param="chi0",
        region="north",
        dataset_sets=["baseline", "baseline_plus_litebird"],
    )

    assert (out_dir / "snr_grid_north_omega0THz_vs_chi0.png").exists()
    npz = np.load(out_dir / "snr_grid_arrays_north_omega0THz_vs_chi0.npz")
    assert "baseline_snr" in npz
    assert "baseline_plus_litebird_snr" in npz
    assert (out_dir / "snr_grid_table_north_omega0THz_vs_chi0.csv").exists()
