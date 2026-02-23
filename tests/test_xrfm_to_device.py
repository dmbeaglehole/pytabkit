"""Test that xRFM models can be moved between devices with .to()."""
import numpy as np
import pytest
import torch

from pytabkit import XRFM_D_Regressor


@pytest.fixture
def fitted_xrfm():
    """Train a small xRFM regressor on CPU."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + 0.1 * rng.standard_normal(200)).astype(np.float32)
    model = XRFM_D_Regressor(device="cpu", iters=2)
    model.fit(X, y)
    return model, X


def test_to_cpu_predict(fitted_xrfm):
    """Predictions should work after .to('cpu')."""
    model, X = fitted_xrfm
    model.to("cpu")
    preds = model.predict(X[:10])
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_to_cuda_predict(fitted_xrfm):
    """Predictions should work after moving a CPU-fitted model to CUDA."""
    model, X = fitted_xrfm
    model.to("cuda")
    preds = model.predict(X[:10])
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_roundtrip_cpu_cuda_cpu(fitted_xrfm):
    """Predictions should be consistent after CPU -> CUDA -> CPU roundtrip."""
    model, X = fitted_xrfm
    preds_before = model.predict(X[:10])

    model.to("cuda")
    preds_cuda = model.predict(X[:10])

    model.to("cpu")
    preds_after = model.predict(X[:10])

    np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)
    assert np.isfinite(preds_cuda).all()
