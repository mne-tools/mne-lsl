from .._typing import ScalarArray as ScalarArray

class WrapStdOut:
    """Dynamically wrap to sys.stdout.

    This makes packages that monkey-patch sys.stdout (e.g.doctest,
    sphinx-gallery) work properly.
    """

    def __getattr__(self, name): ...

def find_events(
    data: ScalarArray,
    first_samp: int,
    verbose: bool | str | int | None = None,
    output: str = None,
    consecutive: bool | str = None,
    min_samples: float = None,
    mask: int | None = None,
    uint_cast: bool = None,
    mask_type: str = None,
    initial_event: bool = None,
    ch_name: str = None,
):
    """Compatibility function for older MNE versions.

    To be dropped when MNE 1.6 is the minimum supported version.
    """
