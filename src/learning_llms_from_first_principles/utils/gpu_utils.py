import torch


def get_device() -> torch.device:
    """
    Determines the best available device for PyTorch in the order:
    CUDA -> TPU -> MPS -> CPU.

    Returns:
        torch.device: The best available device.
    """
    # 1. Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return torch.device("cuda")

    # 2. Check for TPU (requires torch_xla)
    try:
        import torch_xla.core.xla_model as xm

        # If this import succeeds and a device is found, return it
        return xm.xla_device()
    except ImportError:
        pass

    # 3. Check for MPS (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        return torch.device("mps")

    # 4. Default to CPU
    return torch.device("cpu")
