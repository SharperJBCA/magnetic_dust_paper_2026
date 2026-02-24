from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mde_pipeline.fisher.grid_workflow import _build_jobs_from_grid_section


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
