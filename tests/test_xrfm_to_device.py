"""Test that xRFM and TabM models can be moved between devices with .to()."""
import numpy as np
import pytest
import torch
from unittest.mock import patch

from pytabkit import XRFM_D_Regressor
from pytabkit.models.sklearn.sklearn_interfaces import TabM_D_Regressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_xrfm():
    """Train a small xRFM regressor on CPU."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + 0.1 * rng.standard_normal(200)).astype(np.float32)
    model = XRFM_D_Regressor(device="cpu", iters=2)
    model.fit(X, y)
    return model, X


@pytest.fixture
def fitted_tabm():
    """Train a small TabM regressor on CPU."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 10)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + 0.1 * rng.standard_normal(200)).astype(np.float32)
    model = TabM_D_Regressor(
        device="cpu", tabm_k=2, num_emb_type="pwl",
        arch_type="tabm-mini", num_emb_n_bins=2,
    )
    model.fit(X, y)
    return model, X


# ---------------------------------------------------------------------------
# xRFM tests
# ---------------------------------------------------------------------------

def test_xrfm_to_cpu_predict(fitted_xrfm):
    """Predictions should work after .to('cpu')."""
    model, X = fitted_xrfm
    model.to("cpu")
    preds = model.predict(X[:10])
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_xrfm_to_cuda_predict(fitted_xrfm):
    """Predictions should work after moving a CPU-fitted model to CUDA."""
    model, X = fitted_xrfm
    model.to("cuda")
    preds = model.predict(X[:10])
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_xrfm_roundtrip_cpu_cuda_cpu(fitted_xrfm):
    """Predictions should be consistent after CPU -> CUDA -> CPU roundtrip."""
    model, X = fitted_xrfm
    preds_before = model.predict(X[:10])

    model.to("cuda")
    preds_cuda = model.predict(X[:10])

    model.to("cpu")
    preds_after = model.predict(X[:10])

    np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)
    assert np.isfinite(preds_cuda).all()


# ---------------------------------------------------------------------------
# TabM tests
# ---------------------------------------------------------------------------

def test_tabm_to_cpu_predict(fitted_tabm):
    """Predictions should work after .to('cpu')."""
    model, X = fitted_tabm
    model.to("cpu")
    preds = model.predict(X[:10])
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tabm_to_cuda_predict(fitted_tabm):
    """Predictions should work after moving a CPU-fitted model to CUDA."""
    model, X = fitted_tabm
    model.to("cuda")
    preds = model.predict(X[:10])
    assert preds.shape == (10,)
    assert np.isfinite(preds).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tabm_roundtrip_cpu_cuda_cpu(fitted_tabm):
    """Predictions should be consistent after CPU -> CUDA -> CPU roundtrip."""
    model, X = fitted_tabm
    preds_before = model.predict(X[:10])

    model.to("cuda")
    preds_cuda = model.predict(X[:10])

    model.to("cpu")
    preds_after = model.predict(X[:10])

    np.testing.assert_allclose(preds_before, preds_after, atol=1e-4)
    assert np.isfinite(preds_cuda).all()


# ---------------------------------------------------------------------------
# CUDA-unavailable fallback tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_xrfm_cuda_to_cpu_when_cuda_becomes_unavailable(fitted_xrfm):
    """Model trained with CUDA available can predict on CPU after .to('cpu'),
    even when CUDA is no longer available."""
    model, X = fitted_xrfm
    # Move to CUDA first, then back to CPU
    model.to("cuda")
    model.to("cpu")

    # Mock CUDA as unavailable and verify CPU prediction still works
    with patch("torch.cuda.is_available", return_value=False):
        preds = model.predict(X[:10])
        assert preds.shape == (10,)
        assert np.isfinite(preds).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tabm_cuda_to_cpu_when_cuda_becomes_unavailable(fitted_tabm):
    """Model trained with CUDA available can predict on CPU after .to('cpu'),
    even when CUDA is no longer available."""
    model, X = fitted_tabm
    # Move to CUDA first, then back to CPU
    model.to("cuda")
    model.to("cpu")

    # Mock CUDA as unavailable and verify CPU prediction still works
    with patch("torch.cuda.is_available", return_value=False):
        preds = model.predict(X[:10])
        assert preds.shape == (10,)
        assert np.isfinite(preds).all()
