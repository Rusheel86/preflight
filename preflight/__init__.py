"""preflight — pre-flight checks for PyTorch pipelines."""

from preflight.registry import CheckResult, Registry, Severity
from preflight.runner import run_checks

__version__ = "0.1.0"
__all__ = ["Registry", "CheckResult", "Severity", "run_checks"]
