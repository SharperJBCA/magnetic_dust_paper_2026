from types import SimpleNamespace

import numpy as np

from mde_pipeline.fitting.data_types import ComponentSpec, FitData, Model


def _make_template(npix: int) -> SimpleNamespace:
    return SimpleNamespace(
        I=np.ones(npix),
        Q=np.ones(npix) * 2.0,
        U=np.ones(npix) * 3.0,
        m=SimpleNamespace(has_pol=True),
    )


def test_model_predict_ignores_component_stokes_not_in_fit_stokes():
    npix = 4
    fitdata = FitData(
        data=np.zeros((1, 2 * npix)),
        ivar=np.ones((1, 2 * npix)),
        stokes=["Q", "U"],
        frequencies_ghz=np.array([30.0]),
        pixels=np.arange(npix),
        calerror=np.array([0.0]),
        map_names=["map30"],
    )

    spec = ComponentSpec(
        name="sync",
        cls_name="SynchPowerLaw",
        template_name="t_sync",
        params_map={"A": "A_sync", "beta": "beta_sync", "nu0_ghz": "nu0_sync"},
        stokes=["I", "Q", "U"],
        fixed_params={},
        priors=None,
    )

    model = Model(components=[spec], templates={"t_sync": _make_template(npix)}, stokes_order=fitdata.stokes)
    pred = model.predict(
        fitdata,
        {"A_sync": 1.0, "beta_sync": 0.0, "nu0_sync": 30.0},
    )

    assert pred.shape == (1, 2 * npix)
    np.testing.assert_allclose(pred[0, :npix], 2.0)
    np.testing.assert_allclose(pred[0, npix:], 3.0)
