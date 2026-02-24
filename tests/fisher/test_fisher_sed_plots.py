from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mde_pipeline.fisher.run import _collect_parameter_bounds, _draw_valid_posterior_samples, _write_fisher_sed_plots
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


def test_write_fisher_sed_plots_skips_spinningdust_in_polarized_legend(tmp_path, monkeypatch):
    spec_spin = SimpleNamespace(name="spinningdust", template_name="tmpl", fixed_params={}, params_map={})
    spec_sync = SimpleNamespace(name="synchrotron", template_name="tmpl", fixed_params={}, params_map={})

    comp_spin = SimpleNamespace(stokes_out=("I",))
    comp_sync = SimpleNamespace(stokes_out=("I", "Q", "U"))
    model = SimpleNamespace(
        _comps=[(spec_spin, comp_spin), (spec_sync, comp_sync)],
        templates={"tmpl": SimpleNamespace(I=np.ones(8, dtype=float))},
    )
    fitdata = SimpleNamespace(
        frequencies_ghz=np.array([20.0, 40.0], dtype=float),
        map_names=["test_map_20", "test_map_40"],
    )

    def _fake_eval(spec, _comp, _template, nu_ghz, _params):
        if spec.name == "spinningdust":
            return 5.0 + 0.01 * nu_ghz, 7.0 + 0.02 * nu_ghz
        return 2.0 + 0.01 * nu_ghz, 3.0 + 0.01 * nu_ghz

    captured = {}
    original_savefig = _write_fisher_sed_plots.__globals__["plt"].savefig

    def _capture_savefig(path, *args, **kwargs):
        labels = _write_fisher_sed_plots.__globals__["plt"].gca().get_legend_handles_labels()[1]
        captured[Path(path).name] = labels
        return original_savefig(path, *args, **kwargs)

    monkeypatch.setattr("mde_pipeline.fisher.run._eval_component_region_mean", _fake_eval)
    monkeypatch.setattr("mde_pipeline.fisher.run.plt.savefig", _capture_savefig)

    _write_fisher_sed_plots(
        reg_out=tmp_path,
        model=model,
        fitdata=fitdata,
        params0={"amp": 1.0},
        cov=np.array([[0.05]], dtype=float),
        param_names=["amp"],
    )

    polarized_labels = captured["sed_polarized.png"]
    assert all("spinningdust" not in label for label in polarized_labels)
    assert any("synchrotron" in label for label in polarized_labels)
