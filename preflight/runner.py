from __future__ import annotations

from typing import Any

from preflight.registry import CheckResult, get_checks


def run_checks(
    dataloader: Any,
    model: Any | None = None,
    loss_fn: Any | None = None,
    config: dict | None = None,
) -> list[CheckResult]:
    """Run all registered checks and return results."""
    import preflight.checks.data  # noqa: F401
    import preflight.checks.resources  # noqa: F401
    import preflight.checks.splits  # noqa: F401

    if model is not None:
        import preflight.checks.model  # noqa: F401

    results: list[CheckResult] = []
    cfg = config or {}

    for check_fn in get_checks():
        try:
            result = check_fn(
                dataloader=dataloader,
                model=model,
                loss_fn=loss_fn,
                config=cfg,
            )
            if result is not None:
                if isinstance(result, list):
                    results.extend(result)
                else:
                    results.append(result)
        except Exception as exc:
            from preflight.registry import Severity

            results.append(
                CheckResult(
                    name=getattr(check_fn, "__name__", "unknown"),
                    severity=Severity.WARN,
                    passed=False,
                    message=f"Check raised an unexpected error: {exc}",
                )
            )

    return results
