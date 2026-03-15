from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Callable


class Severity(Enum):
    FATAL = "fatal"
    WARN = "warn"
    INFO = "info"


@dataclasses.dataclass
class CheckResult:
    name: str
    severity: Severity
    passed: bool
    message: str
    fix_hint: str | None = None


_CHECKS: list[Callable] = []


def register(fn: Callable) -> Callable:
    """Decorator to register a check function."""
    _CHECKS.append(fn)
    return fn


def get_checks() -> list[Callable]:
    return list(_CHECKS)


class Registry:
    """Convenience class for external plugin registration."""

    @staticmethod
    def register_check(fn: Callable) -> Callable:
        return register(fn)
