# Changelog

All notable changes to preflight will be documented here.

## [Unreleased]

## [0.1.0] - 2026-03-15

### Added
- 10 core pre-flight checks: NaN/Inf detection, normalisation sanity,
  channel ordering, label leakage, split sizes, VRAM estimation,
  class imbalance, shape mismatch, gradient check
- `.preflight.toml` config file support
- Rich terminal output with severity tiers (FATAL / WARN / INFO)
- JSON output mode (`--format json`) for CI integration
- `preflight run` and `preflight checks` CLI commands
- Fix hints for all failing checks
