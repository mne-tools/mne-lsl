import numpy as np

from .. import io
from ...triggers import TriggerDef
from ... import logger


def merge_events(trigger_file, events, rawfile_in, rawfile_out):
    """
    Merge different events. Can be also used to simply change event values.

    Parameters
    ----------
    trigger_file : str
        The absolute path to the trigger file.
    events : dict
        dict(LABEL_MERGED:[LABEL1, LABEL2, ...]).
    rawfile_in : str
        The absolute path to the .fif file to modify.
    rawfile_out : str
        The absolute path to save the new .fif file.
    """
    tdef = TriggerDef(trigger_file)
    raw, eve = io.read_raw_fif(rawfile_in)

    logger.info('=== Before merging ===')
    notfounds = []
    for key in np.unique(eve[:, 2]):
        if key in tdef.by_value:
            logger.info(
                f'{tdef.by_value[key]}: {len(np.where(eve[:, 2] == key)[0])} events')
        else:
            logger.info(f'{key}: { len(np.where(eve[:, 2] == key)[0])} events')
            notfounds.append(key)
    if notfounds:
        for key in notfounds:
            logger.warning(
                f'Key {key} was not found in the definition file.')

    for key in events:
        ev_src = events[key]
        ev_out = tdef.by_name[key]
        x = []
        for e in ev_src:
            x.append(np.where(eve[:, 2] == tdef.by_name[e])[0])
        eve[np.concatenate(x), 2] = ev_out

    # sanity check
    dups = np.where(0 == np.diff(eve[:, 0]))[0]
    assert len(dups) == 0

    # reset trigger channel
    raw._data[0] *= 0
    raw.add_events(eve, 'TRIGGER')
    raw.save(rawfile_out, overwrite=True)

    logger.info('=== After merging ===')
    for key in np.unique(eve[:, 2]):
        if key in tdef.by_value:
            logger.info
            (f'{tdef.by_value[key]}: {len(np.where(eve[:, 2] == key)[0])} events')
        else:
            logger.info(
                f'{key}:  {len(np.where(eve[:, 2] == key)[0])} events')
