from __future__ import annotations

import hashlib
from typing import Any

import torch

from preflight.registry import CheckResult, Severity, register


def _hash_tensor(t: torch.Tensor) -> str:
    raw = t.detach().cpu().numpy().tobytes()
    return hashlib.sha256(raw[:512]).hexdigest()[:16]  # truncate for speed


@register
def check_label_leakage(
    dataloader: Any,
    model: Any = None,
    loss_fn: Any = None,
    config: dict = None,
) -> CheckResult:
    """FATAL: detect sample overlap between train and validation dataloaders."""
    config = config or {}
    val_loader = config.get("val_dataloader")

    if val_loader is None:
        return CheckResult(
            name="label_leakage",
            severity=Severity.FATAL,
            passed=True,
            message=(
                "No val_dataloader provided — leakage check skipped. "
                "Pass via config={'val_dataloader': ...}."
            ),
        )

    n_batches = config.get("leakage_sample_batches", 20)

    train_hashes: set = set()
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        tensors = batch if isinstance(batch, (list, tuple)) else [batch]
        for t in tensors:
            if isinstance(t, torch.Tensor):
                train_hashes.add(_hash_tensor(t))
                break

    overlap = 0
    val_total = 0
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        tensors = batch if isinstance(batch, (list, tuple)) else [batch]
        for t in tensors:
            if isinstance(t, torch.Tensor):
                val_total += 1
                if _hash_tensor(t) in train_hashes:
                    overlap += 1
                break

    if overlap > 0:
        pct = overlap / max(val_total, 1) * 100
        return CheckResult(
            name="label_leakage",
            severity=Severity.FATAL,
            passed=False,
            message=(
                f"Found {overlap}/{val_total} val samples ({pct:.1f}%) "
                "that also appear in train set."
            ),
            fix_hint="Check your split logic. Ensure no sample appears in both train and val/test.",
        )

    return CheckResult(
        name="label_leakage",
        severity=Severity.FATAL,
        passed=True,
        message=f"No train/val overlap detected in {n_batches} sampled batches.",
    )


@register
def check_split_sizes(
    dataloader: Any,
    model: Any = None,
    loss_fn: Any = None,
    config: dict = None,
) -> CheckResult:
    """INFO: warn on degenerate split sizes."""
    config = config or {}
    val_loader = config.get("val_dataloader")

    train_size = len(dataloader.dataset) if hasattr(dataloader, "dataset") else None
    val_size = len(val_loader.dataset) if (val_loader and hasattr(val_loader, "dataset")) else None

    if train_size is None:
        return CheckResult(
            name="split_sizes",
            severity=Severity.INFO,
            passed=True,
            message="Cannot determine dataset size — split size check skipped.",
        )

    issues = []
    if train_size == 0:
        issues.append("train set is empty")
    if val_size is not None and val_size == 0:
        issues.append("val set is empty")
    if val_size is not None and train_size > 0:
        total = train_size + val_size
        val_pct = val_size / total * 100
        if val_pct < 5:
            issues.append(f"val set is only {val_pct:.1f}% of data (recommend ≥10%)")

    if issues:
        return CheckResult(
            name="split_sizes",
            severity=Severity.INFO,
            passed=False,
            message="Split size issues: " + "; ".join(issues),
            fix_hint="Typical split: 80% train / 10% val / 10% test.",
        )

    msg = f"train={train_size} samples"
    if val_size is not None:
        msg += f", val={val_size} samples"
    return CheckResult(
        name="split_sizes",
        severity=Severity.INFO,
        passed=True,
        message=msg,
    )
