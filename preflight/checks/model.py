from __future__ import annotations

from typing import Any

import torch

from preflight.registry import CheckResult, Severity, register


@register
def check_shape_mismatch(
    dataloader: Any,
    model: Any = None,
    loss_fn: Any = None,
    config: dict = None,
) -> CheckResult:
    """FATAL: verify dataset output shape is compatible with model input."""
    if model is None:
        return CheckResult(
            name="shape_mismatch",
            severity=Severity.FATAL,
            passed=True,
            message="No model provided — shape mismatch check skipped.",
        )

    try:
        batch = next(iter(dataloader))
    except Exception as exc:
        return CheckResult(
            name="shape_mismatch",
            severity=Severity.FATAL,
            passed=False,
            message=f"Could not get a batch from dataloader: {exc}",
        )

    inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
    if not isinstance(inputs, torch.Tensor):
        return CheckResult(
            name="shape_mismatch",
            severity=Severity.INFO,
            passed=True,
            message="First batch item is not a tensor — shape check skipped.",
        )

    try:
        with torch.no_grad():
            _ = model(inputs[:1])
        return CheckResult(
            name="shape_mismatch",
            severity=Severity.FATAL,
            passed=True,
            message=f"Model accepted input shape {tuple(inputs.shape[1:])} without error.",
        )
    except Exception as exc:
        return CheckResult(
            name="shape_mismatch",
            severity=Severity.FATAL,
            passed=False,
            message=f"Model forward pass failed with data shape {tuple(inputs.shape)}: {exc}",
            fix_hint=(
                "Check that your DataLoader output shape matches "
                "the model's expected input dimensions."
            ),
        )


@register
def check_gradients(
    dataloader: Any,
    model: Any = None,
    loss_fn: Any = None,
    config: dict = None,
) -> CheckResult:
    """FATAL: run one backward pass and check for zero/exploding gradients."""
    if model is None or loss_fn is None:
        return CheckResult(
            name="gradient_check",
            severity=Severity.FATAL,
            passed=True,
            message="No model+loss provided — gradient check skipped. Pass loss_fn= to enable.",
        )

    try:
        batch = next(iter(dataloader))
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        targets = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None

        model.zero_grad()
        with torch.enable_grad():
            outputs = model(inputs[:2])
            if targets is not None:
                loss = loss_fn(outputs, targets[:2])
            else:
                loss = outputs.mean()
            loss.backward()

        zero_grads = []
        exploding_grads = []
        none_grads = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    none_grads.append(name)
                elif param.grad.norm().item() == 0.0:
                    zero_grads.append(name)
                elif param.grad.norm().item() > 1000:
                    exploding_grads.append(name)

        issues = []
        if none_grads:
            issues.append(
                f"{len(none_grads)} params have no gradient (dead layers?): {none_grads[:3]}"
            )
        if zero_grads:
            issues.append(f"{len(zero_grads)} params have zero gradient: {zero_grads[:3]}")
        if exploding_grads:
            issues.append(
                f"{len(exploding_grads)} params have exploding gradient "
                f"(norm>1000): {exploding_grads[:3]}"
            )

        if issues:
            return CheckResult(
                name="gradient_check",
                severity=Severity.FATAL,
                passed=False,
                message="Gradient issues detected: " + "; ".join(issues),
                fix_hint="Check requires_grad flags, model architecture, and learning rate.",
            )

        return CheckResult(
            name="gradient_check",
            severity=Severity.FATAL,
            passed=True,
            message="One backward pass completed. All gradients look healthy.",
        )

    except Exception as exc:
        return CheckResult(
            name="gradient_check",
            severity=Severity.FATAL,
            passed=False,
            message=f"Backward pass failed: {exc}",
        )
