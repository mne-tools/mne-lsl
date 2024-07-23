from __future__ import annotations

import hashlib
import platform
from typing import TYPE_CHECKING

import numpy as np
from mne.utils import assert_object_equal
from numpy.testing import assert_allclose, assert_array_equal

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Union

    from mne import Info
    from mne.io import BaseRaw

    from .._typing import ScalarArray


def sha256sum(fname: Union[str, Path]) -> str:
    """Efficiently hash a file."""
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(fname, "rb", buffering=0) as file:
        while n := file.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def match_stream_and_raw_data(data: ScalarArray, raw: BaseRaw) -> None:
    """Check if the data array is part of the provided raw."""
    if "Samples" in raw.ch_names:
        # the stream was emitted from a file with the samples idx in a channel,
        # thus we match the stream and raw data based on this sample idx.
        # /!\ in data, the sample idx does not necessarily increase by 1 because of
        # potential loop in the player.
        ch = raw.ch_names.index("Samples")
        start = data[ch, :][0]
        if start != int(start):
            idx = raw.get_data(picks="Samples").squeeze()
            raise RuntimeError(
                f"Could not cast the stream sample idx channel to int. Start '{start}' "
                f"should be an integer. Sample channel in raw {idx} vs stream "
                f"{data[ch, :]}."
            )
        start = int(start)
    else:
        # the stream was emitted from a file without the sample idx in a channel, thus
        # we match the stream and raw data based on the first (n_channels,) samples.
        good = np.isclose(raw[:][0], data[:, :1], atol=1e-10, rtol=1e-8).all(axis=0)
        good = np.where(good)[0]
        if len(good) != 1:
            raise RuntimeError(
                f"Could not match stream and raw data (found {len(good)} options)."
            )
        start = int(good[0])
        del good
    stop = start + data.shape[1]
    n_fetch = 1
    if stop <= raw.times.size:
        raw_data = raw[:, start:stop][0]
    else:
        raw_data = raw[:, start:][0]
        while raw_data.shape[1] != data.shape[1]:
            if raw.times.size <= data.shape[1] - raw_data.shape[1]:
                raw_data = np.hstack((raw_data, raw[:, :][0]))
            else:
                raw_data = np.hstack(
                    (raw_data, raw[:, : data.shape[1] - raw_data.shape[1]][0])
                )
        n_fetch += 1

    if "Samples" in raw.ch_names:
        data_samp_nums = data[ch, :]
        raw_samp_nums = raw_data[ch, :]
        raw_deltas = np.diff(raw_samp_nums)
        raw_delta_idx = np.where(raw_deltas != 1)[0]
        data_deltas = np.diff(data_samp_nums)
        data_delta_idx = np.where(data_deltas != 1)[0]
        raw_deltas = np.round(raw_deltas).astype(int)
        data_deltas = np.round(data_deltas).astype(int)
        assert_array_equal(
            data_samp_nums,
            raw_samp_nums,
            err_msg=(
                f"Samples mismatch, after {n_fetch} fetch(es), with deltas:\n"
                f"  Raw: {raw_delta_idx} ({raw_deltas[raw_delta_idx]})\n"
                f"  Stream: {data_delta_idx} ({data_deltas[data_delta_idx]})"
            ),
        )
    # TODO: Fix the tolerance, on macOS we get differences like
    # -0.030293 vs -0.03031, -0.030286 vs -0.030313, ...
    atol = 0.0001 if platform.system() == "Darwin" else 0.0
    assert_allclose(
        data,
        raw_data,
        rtol=0.001,
        atol=atol,
        err_msg=f"data mismatch, after {n_fetch} fetch(es).",
    )


def compare_infos(info1: Info, info2: Info) -> None:
    """Check that 2 infos are similar, even if some minor attribute deviate."""
    assert info1["ch_names"] == info2["ch_names"]
    assert info1["highpass"] == info2["highpass"]
    assert info2["lowpass"] == info2["lowpass"]
    assert info1["sfreq"] == info2["sfreq"]
    # projectors
    assert len(info1["projs"]) == len(info2["projs"])
    projs1 = sorted(info1["projs"], key=lambda x: x["desc"])
    projs2 = sorted(info2["projs"], key=lambda x: x["desc"])
    for proj1, proj2 in zip(projs1, projs2):
        assert proj1["desc"] == proj2["desc"]
        assert proj1["kind"] == proj2["kind"]
        assert proj1["data"]["nrow"] == proj2["data"]["nrow"]
        assert proj1["data"]["ncol"] == proj2["data"]["ncol"]
        assert proj1["data"]["row_names"] == proj2["data"]["row_names"]
        assert proj1["data"]["col_names"] == proj2["data"]["col_names"]
        assert_allclose(proj1["data"]["data"], proj2["data"]["data"])
    # digitization
    if info1["dig"] is not None and info2["dig"] is not None:
        assert len(info1["dig"]) == len(info2["dig"])
        digs1 = sorted(info1["dig"], key=lambda x: (x["kind"], x["ident"]))
        digs2 = sorted(info2["dig"], key=lambda x: (x["kind"], x["ident"]))
        for dig1, dig2 in zip(digs1, digs2):
            assert dig1["kind"] == dig2["kind"]
            assert dig1["ident"] == dig2["ident"]
            assert dig1["coord_frame"] == dig2["coord_frame"]
            assert_allclose(dig1["r"], dig2["r"])
    elif info1["dig"] is None and info2["dig"] is None:
        pass
    else:
        raise AssertionError
    # channel information, strict order
    chs1 = [
        {
            key: value
            for key, value in elt.items()
            if key
            in (
                "kind",
                "coil_type",
                "loc",
                "unit",
                "unit_mul",
                "ch_name",
                "coord_frame",
            )
        }
        for elt in info1["chs"]
    ]
    chs2 = [
        {
            key: value
            for key, value in elt.items()
            if key
            in (
                "kind",
                "coil_type",
                "loc",
                "unit",
                "unit_mul",
                "ch_name",
                "coord_frame",
            )
        }
        for elt in info2["chs"]
    ]
    assert_object_equal(chs1, chs2)
    range_cal1 = [elt["range"] * elt["cal"] for elt in info1["chs"]]
    range_cal2 = [elt["range"] * elt["cal"] for elt in info2["chs"]]
    assert_allclose(range_cal1, range_cal2)
