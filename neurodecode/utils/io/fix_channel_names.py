from __future__ import print_function, division

from neurodecode import logger
import neurodecode.utils.io as io

#----------------------------------------------------------------------
def fix_channel_names(fif_dir, new_channel_names):
    '''
    Change the channel names of all fif files in a given directory.

    Parameters
    ----------
    fif_dir : str
         The path to the directory containing fif files
    new_channel_names : list
        The list of the new channel names
    '''

    flist = []
    for f in io.get_file_list(fif_dir):
        if io.parse_path(f).ext == 'fif':
            flist.append(f)

    if len(flist) > 0:
        io.make_dirs('%s/corrected' % fif_dir)
        
        for f in io.get_file_list(fif_dir):
            logger.info('\nLoading %s' % f)
            p = io.parse_path(f)
            
            if p.ext == 'fif':
                raw, eve = io.load_fif_raw(f)
                
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

#----------------------------------------------------------------------
if __name__ == '__main__':
    
    import sys
    
    if len(sys.argv) > 3:
        raise IOError("Too many arguments provided. Max is 2: fif_dir; new_channel_names")
    
    if len(sys.argv) == 3:
        fif_dir = sys.argv[1]
        new_channel_names =  sys.argv[2]
    
    if len(sys.argv) == 2:
        fif_dir = sys.argv[1]
        new_channel_names = float(input('New channel names (list)? \n>> '))
        
    if len(sys.argv) == 1:
        fif_dir = input('Directory path containing the file files? \n>> ')
        new_channel_names = list(input('New channel names (list)? \n>> '))
        
    fix_channel_names(fif_dir, new_channel_names)

    print('Finished.')
