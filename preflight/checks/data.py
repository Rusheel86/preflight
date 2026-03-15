from __future__ import annotations

from typing import Any

import torch

from preflight.registry import CheckResult, Severity, register


@register
def check_nan_inf(
    dataloader: Any,
    model: Any = None,
    loss_fn: Any = None,
    config: dict = None,
) -> CheckResult:
    """FATAL: detect NaN or Inf values in dataset samples."""
    config = config or {}
    n_batches = config.get("nan_sample_batches", 10)
    nan_count = 0
    inf_count = 0
    total = 0

    try:
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            tensors = batch if isinstance(batch, (list, tuple)) else [batch]
            for t in tensors:
                if not isinstance(t, torch.Tensor):
                    continue
                total += t.numel()
                nan_count += int(torch.isnan(t).sum().item())
                inf_count += int(torch.isinf(t).sum().item())
    except Exception as exc:
        return CheckResult(
            name="nan_inf_detection",
            severity=Severity.WARN,
            passed=False,
            message=f"Could not iterate dataloader: {exc}",
        )

    if nan_count > 0 or inf_count > 0:
        pct = (nan_count + inf_count) / max(total, 1) * 100
        return CheckResult(
            name="nan_inf_detection",
            severity=Severity.FATAL,
            passed=False,
            message=(
                f"Found {nan_count} NaN and {inf_count} Inf values "
                f"({pct:.2f}% of sampled data)."
            ),
            fix_hint=(
                "Common causes: dividing by zero std, log(0), or missing values imputed as NaN. "
                "Check your normalization transforms."
            ),
        )

    return CheckResult(
        name="nan_inf_detection",
        severity=Severity.FATAL,
        passed=True,
        message=f"No NaN or Inf values found in {n_batches} sampled batches.",
    )


@register
def check_normalisation(
    dataloader: Any,
    model: Any = None,
    loss_fn: Any = None,
    config: dict = None,
) -> CheckResult:
    """WARN: check that tensor values look normalised."""
    config = config or {}
    n_batches = config.get("norm_sample_batches", 5)
    means = []
    stds = []

    try:
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            tensors = batch if isinstance(batch, (list, tuple)) else [batch]
            for t in tensors:
                if not isinstance(t, torch.Tensor) or t.dtype not in (torch.float32, torch.float64):
                    continue
                means.append(t.float().mean().item())
                stds.append(t.float().std().item())
    except Exception:
        return CheckResult(
            name="normalisation_sanity",
            severity=Severity.WARN,
            passed=True,
            message="Could not sample batches for normalisation check — skipped.",
        )

    if not means:
        return CheckResult(
            name="normalisation_sanity",
            severity=Severity.INFO,
            passed=True,
            message="No float tensors found to check normalisation.",
        )

    import statistics

    avg_mean = statistics.mean(means)
    avg_std = statistics.mean(stds)

    issues = []
    if not (-3.0 < avg_mean < 3.0):
        issues.append(f"mean={avg_mean:.3f} (expected near 0 for normalised data)")
    if not (0.05 < avg_std < 10.0):
        issues.append(f"std={avg_std:.3f} (expected ~1.0 for normalised data)")
    if avg_mean > 50 or avg_std > 100:
        issues.append("values look unnormalised (raw pixel range?)")

    if issues:
        return CheckResult(
            name="normalisation_sanity",
            severity=Severity.WARN,
            passed=False,
            message="Possible normalisation issue: " + "; ".join(issues),
            fix_hint=(
                f"Computed stats — mean: {avg_mean:.4f}, std: {avg_std:.4f}. "
                "Add torchvision.transforms.Normalize(mean=[...], std=[...]) to your pipeline."
            ),
        )

    return CheckResult(
        name="normalisation_sanity",
        severity=Severity.WARN,
        passed=True,
        message=f"Normalisation looks reasonable (mean={avg_mean:.3f}, std={avg_std:.3f}).",
    )


@register
def check_channel_ordering(
    dataloader: Any,
    model: Any = None,
    loss_fn: Any = None,
    config: dict = None,
) -> CheckResult:
    """WARN: detect likely NHWC tensors when NCHW is expected."""
    try:
        batch = next(iter(dataloader))
    except Exception:
        return CheckResult(
            name="channel_ordering",
            severity=Severity.WARN,
            passed=True,
            message="Could not sample a batch for channel ordering check.",
        )

    tensors = batch if isinstance(batch, (list, tuple)) else [batch]
    for t in tensors:
        if not isinstance(t, torch.Tensor) or t.ndim != 4:
            continue
        B, d1, d2, d3 = t.shape
        # heuristic: if last dim is 1/3/4 and spatial dims are much larger → NHWC
        if d3 in (1, 3, 4) and d1 > 8 and d2 > 8 and d1 > d3 and d2 > d3:
            return CheckResult(
                name="channel_ordering",
                severity=Severity.WARN,
                passed=False,
                message=(
                    f"Tensor shape {tuple(t.shape)} looks like NHWC (channels-last). "
                    "PyTorch conv layers expect NCHW (channels-first)."
                ),
                fix_hint=(
                    "Add .permute(0, 3, 1, 2) to your dataset " "__getitem__ or transform pipeline."
                ),
            )

    return CheckResult(
        name="channel_ordering",
        severity=Severity.WARN,
        passed=True,
        message="Channel ordering looks correct (NCHW).",
    )
