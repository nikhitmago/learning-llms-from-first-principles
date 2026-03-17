from pathlib import Path

from omegaconf import OmegaConf

from learning_llms_from_first_principles import config, data, inference, modules, utils

__all__ = ["config", "data", "inference", "modules", "utils"]

# Register a custom OmegaConf resolver so YAML configs can reference the package root
# Usage in YAML: ${pkg_root:}
_PKG_ROOT = str(Path(__file__).resolve().parent)
OmegaConf.register_new_resolver("pkg_root", lambda: _PKG_ROOT, replace=True)
