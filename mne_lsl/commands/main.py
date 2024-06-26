from __future__ import annotations

import click

from .sys_info import run as sys_info


@click.group()
def run() -> None:  # noqa: D401
    """Main package entry-point."""


run.add_command(sys_info)
