from __future__ import annotations

from typing import Any

import torch

from preflight.registry import CheckResult, Severity, register


@register
def check_vram(
    dataloader: Any,
    model: Any = None,
    loss_fn: Any = None,
    config: dict = None,
) -> CheckResult:
    """WARN: estimate peak VRAM and warn if near GPU limit."""
    if model is None:
        return CheckResult(
            name="vram_estimation",
            severity=Severity.WARN,
            passed=True,
            message="No model provided — VRAM estimation skipped.",
        )

    if not torch.cuda.is_available():
        return CheckResult(
            name="vram_estimation",
            severity=Severity.INFO,
            passed=True,
            message="No CUDA GPU detected — VRAM check skipped.",
        )

    try:
        total_vram = torch.cuda.get_device_properties(0).total_memory
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        # heuristic: params + gradients + optimizer states + activations ≈ 4x
        estimated_peak = param_bytes * 4

        pct = estimated_peak / total_vram * 100
        total_gb = total_vram / 1024**3
        est_gb = estimated_peak / 1024**3

        if pct > 90:
            return CheckResult(
                name="vram_estimation",
                severity=Severity.WARN,
                passed=False,
                message=(
                    f"Estimated peak VRAM {est_gb:.2f} GB is {pct:.0f}% of "
                    f"GPU total {total_gb:.2f} GB. Risk of OOM."
                ),
                fix_hint=(
                    "Consider: gradient checkpointing (torch.utils.checkpoint), "
                    "mixed precision (torch.cuda.amp), or reducing batch size."
                ),
            )

        return CheckResult(
            name="vram_estimation",
            severity=Severity.WARN,
            passed=True,
            message=(
                f"Estimated peak VRAM: {est_gb:.2f} GB / "
                f"{total_gb:.2f} GB available ({pct:.0f}%)."
            ),
        )

    except Exception as exc:
        return CheckResult(
            name="vram_estimation",
            severity=Severity.WARN,
            passed=True,
            message=f"VRAM estimation failed: {exc}",
        )


@register
def check_class_imbalance(
    dataloader: Any,
    model: Any = None,
    loss_fn: Any = None,
    config: dict = None,
) -> CheckResult:
    """WARN: detect severe class imbalance in labels."""
    config = config or {}
    threshold = config.get("imbalance_threshold", 0.1)
    n_batches = config.get("imbalance_sample_batches", 10)

    label_counts: dict = {}

    try:
        for i, batch in enumerate(dataloader):
            if i >= n_batches:
                break
            labels = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
            if labels is None or not isinstance(labels, torch.Tensor):
                continue
            for val in labels.flatten().tolist():
                key = int(val) if isinstance(val, float) and val == int(val) else val
                label_counts[key] = label_counts.get(key, 0) + 1
    except Exception:
        return CheckResult(
            name="class_imbalance",
            severity=Severity.WARN,
            passed=True,
            message="Could not extract labels for imbalance check — skipped.",
        )

    if not label_counts:
        return CheckResult(
            name="class_imbalance",
            severity=Severity.INFO,
            passed=True,
            message="No integer labels found — class imbalance check skipped.",
        )

    total = sum(label_counts.values())
    n_classes = len(label_counts)
    expected_freq = 1.0 / n_classes
    imbalanced = [
        (cls, cnt / total)
        for cls, cnt in label_counts.items()
        if (cnt / total) <= expected_freq * threshold
    ]

    if imbalanced:
        details = ", ".join(f"class {c}: {f*100:.1f}%" for c, f in imbalanced[:5])
        return CheckResult(
            name="class_imbalance",
            severity=Severity.WARN,
            passed=False,
            message=f"Severe imbalance in {len(imbalanced)} class(es): {details}",
            fix_hint=(
                "Consider WeightedRandomSampler or class-weighted loss. "
                "Set config={'imbalance_threshold': 0.05} to adjust sensitivity."
            ),
        )

    dist = ", ".join(f"{c}: {cnt/total*100:.1f}%" for c, cnt in sorted(label_counts.items())[:6])
    return CheckResult(
        name="class_imbalance",
        severity=Severity.WARN,
        passed=True,
        message=f"Class distribution looks balanced. ({dist})",
    )
