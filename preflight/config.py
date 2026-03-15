from __future__ import annotations

from pathlib import Path


def load_config(config_path: Path | None = None) -> dict:
    """Load .preflight.toml from the given path or the current directory."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return {}

    path = config_path or Path(".preflight.toml")
    if path.exists():
        with open(path, "rb") as f:
            return tomllib.load(f)
    return {}
