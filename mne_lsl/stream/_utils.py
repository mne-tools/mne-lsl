from ..utils._checks import check_type


def check_bufsize(bufsize: float) -> None:
    """Check the bufsize argument."""
    check_type(bufsize, ("numeric",), "bufsize")
    if bufsize <= 0:
        raise ValueError(
            "The buffer size 'bufsize' must be a strictly positive number. "
            f"{bufsize} is invalid."
        )
