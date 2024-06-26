from __future__ import annotations

import click

from .player import run as player
from .sys_info import run as sys_info
from .viewer import run as viewer


@click.group()
def run() -> None:  # noqa: D401
    """Main package entry-point."""


run.add_command(player)
run.add_command(sys_info)
run.add_command(viewer)
