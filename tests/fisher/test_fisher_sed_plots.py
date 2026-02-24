from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mde_pipeline.fisher.run import _collect_parameter_bounds, _draw_valid_posterior_samples
from mde_pipeline.fitting.priors import BoundsPrior, JointPrior


def test_collect_parameter_bounds_from_joint_prior_bounds_prior_only():
    gp = JointPrior(
        [
            BoundsPrior({"T_d": (0.0, None), "beta_d": (1.0, 2.5)}),
            BoundsPrior({"T_d": (1.0, 100.0)}),
        ]
    )

    bounds = _collect_parameter_bounds(gp)

    assert bounds["T_d"] == (1.0, 100.0)
    assert bounds["beta_d"] == (1.0, 2.5)


def test_draw_valid_posterior_samples_filters_non_physical_temperature_draws():
    rng = np.random.default_rng(7)
    param_names = ["T_d", "A_d"]
    means = np.array([-10.0, 1.0], dtype=float)
    cov = np.array([[500.0, 0.0], [0.0, 0.1]], dtype=float)

    draws = _draw_valid_posterior_samples(
        means=means,
        cov=cov,
        param_names=param_names,
        bounds_lookup={"T_d": (0.0, None)},
        sample_count=25,
        rng=rng,
        max_attempt_factor=300,
    )

    assert draws.shape[0] == 25
    assert np.all(draws[:, 0] > 0.0)
