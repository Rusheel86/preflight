# Contributing to preflight

## Setup
```bash
git clone https://github.com/Rusheel86/preflight.git
cd preflight
conda create -n preflight-dev python=3.11 -y
conda activate preflight-dev
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pre-commit install
```

## Running tests
```bash
pytest
```

## Adding a new check

1. Add a function to the appropriate file in `preflight/checks/`
2. Decorate it with `@register`
3. Return a `CheckResult` with the correct severity
4. Write two tests: one that passes, one that fails
5. Add an entry to `CHANGELOG.md`

Every check function must accept `(dataloader, model, loss_fn, config)` as keyword arguments even if it only uses some of them.

## Commit style

`fix: correct NaN detection in multi-output dataloaders`
`feat: add duplicate sample detection check`
`docs: add example for custom config`

Sign all commits: `git commit -s -m "your message"`
