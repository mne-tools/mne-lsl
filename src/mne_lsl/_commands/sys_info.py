from __future__ import annotations

import click

from .. import sys_info


@click.command(name="sys-info")
@click.option(
    "--extra",
    help="Display information for optional dependencies.",
    is_flag=True,
)
@click.option(
    "--developer",
    help="Display information for developer dependencies.",
    is_flag=True,
)
def run(extra: bool, developer: bool) -> None:
    """Get the system information."""
    sys_info(extra=extra, developer=developer)
