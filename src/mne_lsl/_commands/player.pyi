from pathlib import Path

from .. import set_log_level as set_log_level
from ..player import PlayerLSL as PlayerLSL

def run(
    fname: Path,
    chunk_size: int,
    n_repeat: int | None,
    name: str,
    annotations: bool,
    verbose: str,
) -> None:
    """Run a Player to mock a real-time stream."""
