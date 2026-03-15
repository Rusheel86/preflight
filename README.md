# preflight

> Pre-flight checks for PyTorch pipelines. Catch silent failures before they waste your GPU.

[![CI](https://github.com/Rusheel86/preflight/actions/workflows/ci.yml/badge.svg)](https://github.com/Rusheel86/preflight/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/preflight-ml)](https://pypi.org/project/preflight-ml/)
[![Python](https://img.shields.io/pypi/pyversions/preflight-ml)](https://pypi.org/project/preflight-ml/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

Most deep learning bugs don't crash your training loop — they silently produce a garbage model.
NaNs in your data, labels leaking between train and val, wrong channel ordering, dead gradients.
You won't know until hours later, after the GPU bill has landed.

**preflight** is a pre-training validation tool you run in 30 seconds before starting any training job.
It's not a linter. It's a pre-flight check — the kind pilots run before the expensive thing takes off.

---

## Install

```bash
pip install preflight-ml
```

## Quickstart

Create a small Python file that exposes your dataloader:

```python
# my_dataloader.py
import torch
from torch.utils.data import DataLoader, TensorDataset

x = torch.randn(200, 3, 224, 224)
y = torch.randint(0, 10, (200,))
dataloader = DataLoader(TensorDataset(x, y), batch_size=32)
```

Run preflight:

```bash
preflight run --dataloader my_dataloader.py
```

Output:

```
preflight — pre-training check report
╭────────────────────────┬──────────┬────────┬──────────────────────────────────────────────────╮
│ Check                  │ Severity │ Status │ Message                                          │
├────────────────────────┼──────────┼────────┼──────────────────────────────────────────────────┤
│ nan_inf_detection      │ FATAL    │ PASS   │ No NaN or Inf values found in 10 sampled batches │
│ normalisation_sanity   │ WARN     │ PASS   │ Normalisation looks reasonable (mean=0.001)      │
│ channel_ordering       │ WARN     │ PASS   │ Channel ordering looks correct (NCHW)            │
│ label_leakage          │ FATAL    │ PASS   │ No val_dataloader provided — skipped             │
│ split_sizes            │ INFO     │ PASS   │ train=200 samples                                │
│ vram_estimation        │ WARN     │ INFO   │ No CUDA GPU detected — skipped                   │
│ class_imbalance        │ WARN     │ PASS   │ Class distribution looks balanced                │
│ shape_mismatch         │ FATAL    │ PASS   │ No model provided — skipped                      │
│ gradient_check         │ FATAL    │ PASS   │ No model+loss provided — skipped                 │
╰────────────────────────┴──────────┴────────┴──────────────────────────────────────────────────╯

  0 fatal  0 warnings  9 passed

Pre-flight passed. Safe to start training.
```

## Checks

preflight runs 10 checks across three severity tiers. A **FATAL** failure exits with code 1 and blocks CI.

| Check | Severity | What it catches |
|---|---|---|
| `nan_inf_detection` | FATAL | NaN or Inf values anywhere in sampled batches |
| `label_leakage` | FATAL | Samples appearing in both train and val sets |
| `shape_mismatch` | FATAL | Dataset output shape incompatible with model input |
| `gradient_check` | FATAL | Zero gradients, dead layers, exploding gradients |
| `normalisation_sanity` | WARN | Data that looks unnormalised (raw pixel values etc.) |
| `channel_ordering` | WARN | NHWC tensors when PyTorch expects NCHW |
| `vram_estimation` | WARN | Estimated peak VRAM exceeds 90% of GPU memory |
| `class_imbalance` | WARN | Severe class imbalance beyond configurable threshold |
| `split_sizes` | INFO | Empty or degenerate train/val splits |
| `duplicate_samples` | INFO | Identical samples within a split |

## With a model

Pass a model file to enable shape, gradient, and VRAM checks:

```python
# my_model.py
import torch.nn as nn
model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 224 * 224, 10))
```

```python
# my_loss.py
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss()
```

```bash
preflight run \
  --dataloader my_dataloader.py \
  --model my_model.py \
  --loss my_loss.py \
  --val-dataloader my_val_dataloader.py
```

## Configuration

Add a `.preflight.toml` to your repo root to configure thresholds and disable checks:

```toml
[thresholds]
imbalance_threshold = 0.05
nan_sample_batches = 20

[checks]
vram_estimation = false

[ignore]
# check = "class_imbalance"
# reason = "intentional: rare event dataset"
```

## CI integration

Add to your GitHub Actions workflow:

```yaml
- name: Install preflight
  run: pip install preflight-ml

- name: Run pre-flight checks
  run: preflight run --dataloader scripts/dataloader.py --format json
```

The `--format json` flag outputs machine-readable results. Exit code is `1` if any FATAL check fails, `0` otherwise.

## List all checks

```bash
preflight checks
```

## What preflight does NOT do

- It does not replace unit tests. Use pytest for code logic.
- It does not guarantee a correct model. Passing preflight is a minimum safety bar, not a certification.
- It does not run your full training loop. Use it as a gate before training starts.
- It does not modify your code unless you pass `--fix`.

## Roadmap

- [ ] `--fix` flag — auto-patch common issues (channel ordering, normalisation)
- [ ] Dataset snapshot + drift detection (`preflight diff baseline.json new_data.pt`)
- [ ] Full dry-run mode (one batch through model + loss + backward)
- [ ] Jupyter magic command (`%load_ext preflight`)
- [ ] `preflight-monai` plugin for medical imaging checks
- [ ] `preflight-sktime` plugin for time series checks

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). New checks are welcome — each one needs a passing test,
a failing test, and a fix hint.

## License

MIT — see [LICENSE](LICENSE).
