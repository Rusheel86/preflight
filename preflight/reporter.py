from __future__ import annotations

import json

from rich import box
from rich.console import Console
from rich.table import Table

from preflight.registry import CheckResult, Severity

console = Console()

SEVERITY_COLORS = {
    Severity.FATAL: "red",
    Severity.WARN: "yellow",
    Severity.INFO: "blue",
}

SEVERITY_LABELS = {
    Severity.FATAL: "FATAL",
    Severity.WARN: "WARN ",
    Severity.INFO: "INFO ",
}


def print_results(results: list[CheckResult], fmt: str = "terminal") -> int:
    """Print results and return exit code (1 if any FATAL failed)."""
    if fmt == "json":
        output = [
            {
                "name": r.name,
                "severity": r.severity.value,
                "passed": r.passed,
                "message": r.message,
                "fix_hint": r.fix_hint,
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))
        fatal_failures = [r for r in results if r.severity == Severity.FATAL and not r.passed]
        return 1 if fatal_failures else 0

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold")
    table.add_column("Check", style="dim", width=24)
    table.add_column("Severity", width=8)
    table.add_column("Status", width=6)
    table.add_column("Message")

    fatal_count: int = 0
    warn_count: int = 0
    pass_count: int = 0

    for r in results:
        color = SEVERITY_COLORS[r.severity]
        sev_label = SEVERITY_LABELS[r.severity]
        status = "[green]PASS[/green]" if r.passed else f"[{color}]FAIL[/{color}]"
        table.add_row(r.name, f"[{color}]{sev_label}[/{color}]", status, r.message)

        if not r.passed:
            if r.severity == Severity.FATAL:
                fatal_count += 1
            elif r.severity == Severity.WARN:
                warn_count += 1
        else:
            pass_count += 1

    console.print()
    console.print("[bold]preflight[/bold] — pre-training check report")
    console.print(table)

    hints = [(r.name, r.fix_hint) for r in results if r.fix_hint and not r.passed]
    if hints:
        console.print("\n[bold yellow]Fix hints:[/bold yellow]")
        for name, hint in hints:
            console.print(f"  [dim]{name}:[/dim] {hint}")

    console.print(
        f"\n  [red]{fatal_count} fatal[/red]  "
        f"[yellow]{warn_count} warnings[/yellow]  "
        f"[green]{pass_count} passed[/green]"
    )

    if fatal_count > 0:
        console.print(
            "\n[red]Pre-flight failed.[/red] Fix fatal issues before training.\n"
            "[dim]Note: passing preflight is a minimum safety bar, "
            "not a guarantee of correct training.[/dim]"
        )
    else:
        console.print("\n[green]Pre-flight passed.[/green] Safe to start training.\n")

    return 1 if fatal_count > 0 else 0
