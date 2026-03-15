import sys
from unittest.mock import patch

from learning_llms_from_first_principles.utils.gpu_utils import get_device


def test_get_device_cuda() -> None:
    """Test that CUDA is preferred if available."""
    with patch("torch.cuda.is_available", return_value=True):
        device = get_device()
        assert device.type == "cuda"


def test_get_device_mps() -> None:
    """Test that MPS is used if CUDA/TPU are unavailable but MPS is available."""
    with patch("torch.cuda.is_available", return_value=False):
        # Ensure TPU import fails
        with patch.dict(sys.modules, {"torch_xla.core.xla_model": None}):
            with patch("torch.backends.mps.is_available", return_value=True):
                device = get_device()
                assert device.type == "mps"


def test_get_device_cpu() -> None:
    """Test fallback to CPU when no accelerators are available."""
    with patch("torch.cuda.is_available", return_value=False):
        with patch.dict(sys.modules, {"torch_xla.core.xla_model": None}):
            with patch("torch.backends.mps.is_available", return_value=False):
                device = get_device()
                assert device.type == "cpu"
