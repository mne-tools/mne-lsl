from __future__ import annotations

from pathlib import Path

import click
import numpy as np

from .. import set_log_level
from ..player import PlayerLSL


@click.command(name="player")
@click.argument("fname", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "-c",
    "--chunk_size",
    help="Number of samples pushed at once via LSL.",
    type=int,
    default=10,
    show_default=True,
)
@click.option(
    "--n_repeat",
    help="Number of times to repeat the file.",
    type=int,
)
@click.option(
    "-n",
    "--name",
    help="Name of the stream to emit.",
    type=str,
    default="MNE-LSL-Player",
    show_default=True,
)
@click.option("--annotations", help="Enable streaming of annotations.", is_flag=True)
@click.option(
    "--verbose",
    help="Verbosity level.",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    default="INFO",
    show_default=True,
)
def run(
    fname: Path,
    chunk_size: int,
    n_repeat: int,
    name: str,
    annotations: bool,
    verbose: str,
) -> None:  # pragma: no cover
    """Run a Player to mock a real-time stream."""
    set_log_level(verbose)
    if n_repeat is None:
        n_repeat = np.inf
    player = PlayerLSL(
        fname=fname,
        chunk_size=chunk_size,
        n_repeat=n_repeat,
        name=name,
        annotations=annotations,
    )
    player.start()
    input(">> Press ENTER to stop replaying data \n")
    player.stop()
