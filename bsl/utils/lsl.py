"""LSL wrapper functions for creating a server and a client."""

import xml.etree.ElementTree as ET

from ..lsl import StreamInlet
from ._checks import check_type


def lsl_channel_list(inlet):
    """Extract the channels name list from the LSL info.

    Parameters
    ----------
    inlet : StreamInlet
        Inlet to extract the channels list from.

    Returns
    -------
    ch_list : list
        List of channels name ``[name1, name2, ... ]``.
    """
    check_type(inlet, (StreamInlet,), item_name="inlet")
    root = ET.fromstring(inlet.get_sinfo().as_xml)
    ch_list = []
    for elt in root.iter("channel"):
        ch_list.append(elt.find("label").text)
    return ch_list
