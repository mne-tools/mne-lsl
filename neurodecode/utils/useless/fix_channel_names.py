from __future__ import print_function, division

import neurodecode.utils.q_common as qc

from neurodecode import logger
from neurodecode.utils.io import parse_path, make_dirs, load_fif_raw, get_file_list

def fix_channel_names(fif_dir, new_channel_names):
    '''
    Change channel names of fif files in a given directory.

    Input
    -----
    @fif_dir: path to fif files
    @new_channel_names: list of new channel names

    Output
    ------
    Modified fif files are saved in fif_dir/corrected/

    Kyuhwa Lee, 2019.
    '''

    flist = []
    for f in get_file_list(fif_dir):
        if parse_path(f).ext == 'fif':
            flist.append(f)

    if len(flist) > 0:
        make_dirs('%s/corrected' % fif_dir)
        for f in get_file_list(fif_dir):
            logger.info('\nLoading %s' % f)
            p = parse_path(f)
            if p.ext == 'fif':
                raw, eve = load_fif_raw(f)
                if len(raw.ch_names) != len(new_channel_names):
                    raise RuntimeError('The number of new channels do not matach that of fif file.')
                raw.info['ch_names'] = new_channel_names
                for ch, new_ch in zip(raw.info['chs'], new_channel_names):
                    ch['ch_name'] = new_ch
                out_fif = '%s/corrected/%s.fif' % (p.dir, p.name)
                logger.info('Exporting to %s' % out_fif)
                raw.save(out_fif)
    else:
        logger.warning('No fif files found in %s' % fif_dir)
