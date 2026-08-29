from __future__ import annotations

import click

from .. import set_log_level


@click.command(name="viewer")
@click.option(
    "--verbose",
    help="Verbosity level.",
    type=click.Choice(["DEBUG", "INFO", "WARNING"]),
    default="INFO",
    show_default=True,
)
def run(verbose: str) -> None:
    """Run the Viewer to inspect LSL streams."""
    # the import is nested to keep 'import mne_lsl' free of any Qt import; the
    # 'mne_lsl.viewer' module is added by the viewer scaffold phase.
    from ..viewer import Viewer

    set_log_level(verbose)
    # raised, not returned: click reports the return value of a command nowhere, thus a
    # shell would always see a 0 exit status however the event loop ended.
    raise SystemExit(Viewer().start())
