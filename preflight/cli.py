from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import click

from preflight.reporter import print_results
from preflight.runner import run_checks


def _load_config(config_path: Path | None = None) -> dict:
    """Load .preflight.toml if present."""
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


def _import_object(script_path: str, var_name: str) -> Any:
    """Import a Python object from a script file."""
    spec = importlib.util.spec_from_file_location("_preflight_user_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, var_name, None)


@click.group()
@click.version_option()
def main() -> None:
    """preflight — pre-flight checks for PyTorch pipelines."""


@main.command()
@click.option(
    "--dataloader",
    "dl_script",
    required=True,
    help="Path to Python file. Must define a variable named 'dataloader'.",
)
@click.option(
    "--model",
    "model_script",
    default=None,
    help="Path to Python file. Must define a variable named 'model'.",
)
@click.option(
    "--loss",
    "loss_script",
    default=None,
    help="Path to Python file. Must define a variable named 'loss_fn'.",
)
@click.option(
    "--val-dataloader",
    "val_dl_script",
    default=None,
    help="Path to Python file. Must define a variable named 'dataloader' (val set).",
)
@click.option(
    "--format",
    "fmt",
    default="terminal",
    type=click.Choice(["terminal", "json"]),
    help="Output format.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(),
    help="Path to .preflight.toml config file.",
)
def run(
    dl_script: str,
    model_script: str | None,
    loss_script: str | None,
    val_dl_script: str | None,
    fmt: str,
    config_path: str | None,
) -> None:
    """Run all pre-flight checks on your dataset and model."""
    cfg = _load_config(Path(config_path) if config_path else None)

    dataloader = _import_object(dl_script, "dataloader")
    if dataloader is None:
        click.echo(f"Error: '{dl_script}' must define a variable named 'dataloader'.", err=True)
        sys.exit(1)

    model = _import_object(model_script, "model") if model_script else None
    loss_fn = _import_object(loss_script, "loss_fn") if loss_script else None

    if val_dl_script:
        cfg["val_dataloader"] = _import_object(val_dl_script, "dataloader")

    results = run_checks(dataloader=dataloader, model=model, loss_fn=loss_fn, config=cfg)
    exit_code = print_results(results, fmt=fmt)
    sys.exit(exit_code)


@main.command()
def checks() -> None:
    """List all registered checks."""
    from rich.console import Console

    import preflight.checks.data  # noqa: F401
    import preflight.checks.model  # noqa: F401
    import preflight.checks.resources  # noqa: F401
    import preflight.checks.splits  # noqa: F401
    from preflight.registry import get_checks

    c = Console()
    c.print("\n[bold]Registered checks:[/bold]")
    for fn in get_checks():
        doc = (fn.__doc__ or "").strip().split("\n")[0]
        c.print(f"  [dim]•[/dim] [cyan]{fn.__name__}[/cyan]  {doc}")
    c.print()
