"""
Function fetching file from URL.
Adapted from MNE:
    https://github.com/mne-tools/mne-python
Adapted from NILEARN:
    https://github.com/nilearn/nilearn
"""

import time
import hashlib
import shutil
from math import log
from pathlib import Path
from mne.utils.progressbar import ProgressBar
from urllib import parse, request
from urllib.error import HTTPError, URLError

from ..utils._logs import logger


def fetch_file(
    url,
    file_name,
    print_destination=True,
    resume=True,
    hash_=None,
    hash_type="md5",
    timeout=30.0,
):
    """
    Load requested file, downloading it if needed or requested.

    Parameters
    ----------
    url : str
        The url of the file to be downloaded.
    file_name : str
        Name, along with the path, of where downloaded file will be saved.
    print_destination : bool
        If true, destination of where file was saved will be printed after
        download finishes.
    resume : bool
        If True, try to resume partially downloaded files.
    hash_ : str | None
        The hash of the file to check. If None, no checking is performed.
    hash_type : str
        The type of hashing to use such as 'md5' or 'sha1'.
    timeout : float
        The URL open timeout.
    """
    if hash_ is not None:
        if not isinstance(hash_, str):
            raise ValueError(
                "Bad hash value given, should be a string. Given: %s."
                % type(hash_)
            )
        if hash_type == "md5" and len(hash_) != 32:
            raise ValueError(
                "Bad hash value given, should be a 32-character string:\n%s"
                % hash_
            )
        if hash_type == "sha1" and len(hash_) != 40:
            raise ValueError(
                "Bad hash value given, should be a 40-character string:\n%s"
                % hash_
            )
        if hash_type not in ["md5", "sha1"]:
            raise ValueError(
                "Unsupported hash type %s.\nSupported: 'md5', 'sha1'."
                % hash_type
            )
    file_name = Path(file_name)
    temp_file_name = file_name.with_suffix(file_name.suffix + ".part")
    scheme = parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise NotImplementedError("Cannot use scheme %r" % (scheme,))
    try:
        # Triage resume
        if not temp_file_name.exists():
            resume = False
        if resume:
            with open(temp_file_name, "rb", buffering=0) as local_file:
                local_file.seek(0, 2)
                initial_size = local_file.tell()
            del local_file
        else:
            initial_size = 0
        _get_http(url, temp_file_name, initial_size, timeout)

        # check hash sum
        if hash_ is not None:
            logger.info("Verifying hash %s.", hash_)
            hashsum = _hashfunc(temp_file_name, hash_type=hash_type)
            if hash_ != hashsum:
                raise RuntimeError(
                    "Hash mismatch for downloaded file %s, "
                    "expected %s but got %s" % (temp_file_name, hash_, hashsum)
                )
        shutil.move(temp_file_name, file_name)
        if print_destination is True:
            logger.info("File saved as %s.\n", file_name)
    except Exception:
        logger.error(
            "Error while fetching file %s. Dataset fetching aborted.", url
        )
        raise


def _hashfunc(fname, block_size=1048576, hash_type="md5"):
    """
    Calculate the hash for a file.

    Parameters
    ----------
    fname : str
        Filename.
    block_size : int
        Block size to use when reading.

    Returns
    -------
    hash_ : str
        The hexadecimal digest of the hash.
    """

    if hash_type == "md5":
        hasher = hashlib.md5()
    elif hash_type == "sha1":
        hasher = hashlib.sha1()
    else:
        raise ValueError("Unsupported hash type. Supported: 'md5', 'sha1'.")
    with open(fname, "rb") as fid:
        while True:
            data = fid.read(block_size)
            if not data:
                break
            hasher.update(data)
    return hasher.hexdigest()


def _get_http(url, temp_file_name, initial_size, timeout):
    """Safely (resume a) download to a file from http(s)."""
    response = None
    extra = ""
    if initial_size > 0:
        logger.debug("  Resuming at %s", initial_size)
        req = request.Request(
            url, headers={"Range": "bytes=%s-" % (initial_size,)}
        )
        try:
            response = request.urlopen(req, timeout=timeout)
            content_range = response.info().get("Content-Range", None)
            if content_range is None or not content_range.startswith(
                "bytes %s-" % (initial_size,)
            ):
                raise IOError("Server does not support resuming")
        except (KeyError, HTTPError, URLError, IOError):
            initial_size = 0
            response = None
        else:
            extra = ", resuming at %s" % (_sizeof_fmt(initial_size),)
    if response is None:
        response = request.urlopen(request.Request(url), timeout=timeout)
    file_size = int(response.headers.get("Content-Length", "0").strip())
    file_size += initial_size
    url = response.geturl()
    logger.info("Downloading %s (%s%s)", url, _sizeof_fmt(file_size), extra)
    del url
    mode = "ab" if initial_size > 0 else "wb"
    progress = ProgressBar(
        file_size,
        initial_size,
        unit="B",
        mesg="Downloading",
        unit_scale=True,
        unit_divisor=1024,
    )
    del file_size
    chunk_size = 8192  # 2 ** 13
    with open(temp_file_name, mode) as local_file:
        while True:
            t0 = time.time()
            chunk = response.read(chunk_size)
            dt = time.time() - t0
            if dt < 0.01:
                chunk_size *= 2
            elif dt > 0.1 and chunk_size > 8192:
                chunk_size = chunk_size // 2
            if not chunk:
                break
            local_file.write(chunk)
            progress.update_with_increment_value(len(chunk))


def _sizeof_fmt(num):
    """
    Turn number of bytes into human-readable str.

    Parameters
    ----------
    num : int
        The number of bytes.

    Returns
    -------
    size : str
        The size in human-readable format.
    """

    units = ["bytes", "kB", "MB", "GB", "TB", "PB"]
    decimals = [0, 0, 1, 2, 2, 2]
    if num > 1:
        exponent = min(int(log(num, 1024)), len(units) - 1)
        quotient = float(num) / 1024**exponent
        unit = units[exponent]
        num_decimals = decimals[exponent]
        format_string = "{0:.%sf} {1}" % (num_decimals)
        return format_string.format(quotient, unit)
    if num == 0:
        return "0 bytes"
    if num == 1:
        return "1 byte"
