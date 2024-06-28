from __future__ import annotations

import click

from .. import set_log_level


@click.command(name="viewer")
@click.option(
    "-s",
    "--stream",
    prompt="LSL stream name",
    help="Name of the stream to plot.",
    type=str,
)
@click.option(
    "--verbose",
    help="Verbosity level.",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    default="INFO",
    show_default=True,
)
def run(stream: str, verbose: str) -> None:  # pragma: no cover
    """Run the StreamViewer."""
    from ..stream_viewer import StreamViewer

    set_log_level(verbose)
    stream_viewer = StreamViewer(stream)
    stream_viewer.start()
