from __future__ import print_function, division

import os
import sys
import mne
import multiprocessing as mp
from builtins import input
from neurodecode import logger
from neurodecode.triggers import TriggerDef
from neurodecode.gui.streams import redirect_stdout_to_queue
import neurodecode.utils.io as io

mne.set_log_level('ERROR')
os.environ['OMP_NUM_THREADS'] = '1' # actually improves performance for multitaper

#-------------------------------------------------------------------------
def check_config(cfg):
    """
    Check if the required parameters are defined in the config module
    
    Parameters
    ----------
    cfg : python.module
        The loaded config module
    """
    critical_vars = {
        'COMMON': [ 'AAA',
                    'BBB',
                    'CCC'],
    }

    # Check the critical variables
    optional_vars = {  
        'COMMON': { 'DDD',
                    'EEE'}, 
        
        # Internal parmameters for the AAA
        'XXX': { 'min': 1, 'max': 40, },
    }
    
    # Check the critical variables
    _check_cfg_mandatory(cfg, critical_vars, 'COMMON')
    
    # Check the optional variables
    _check_cfg_optional(cfg, optional_vars, 'COMMON')
    
    # Check the internal param of AAA
    _check_cfg_selected(cfg, optional_vars, 'AAA')
    
    if getattr(cfg, 'TRIGGER_DEVICE') == None:
        logger.warning('The trigger device is set to None! No events will be saved.')    

    return cfg

#-------------------------------------------------------------------------
def batch_run(cfg_module):
    '''
    Used when launch from the terminal (not GUI)
    
    Parameters
    ----------
    cfg_module : str
        The path to the config module
    '''    
    cfg = io.load_config(cfg_module)
    cfg = check_config(cfg)
    run(cfg, interactive=True)

#-------------------------------------------------------------------------
def run(cfg, state=mp.Value('i', 1), queue=None, logger=logger):
    '''
    Main function used to run the offline protocol.
    
    Parameters
    ----------
    cfg : python.module
        The loaded config module from the corresponding config_offline.py
    queue : mp.Queue
        If not None, redirect sys.stdout to GUI terminal
    logger : logging.logger
        The logger to use
    '''    
    redirect_stdout_to_queue(logger, queue, 'INFO')

    # Load the mapping from int to string for triggers events
    cfg.tdef = TriggerDef(cfg.TRIGGER_FILE)
    
    # Protocol start if equals to 1
    if not state.value:
        sys.exit(-1)
    
    #-------------------------------------
    # ADD YOUR CODE HERE
    #-------------------------------------
    
    # TO train a decoder, look at trainer_mi.py protocol
    
    with state.get_lock():
        state.value = 0

#-------------------------------------------------------------------------
def _check_cfg_optional(cfg, optional_vars, key_var):
    """
    Check that the optional parameters are defined and if not assign them
    
    Parameters
    ----------
    cfg :python.module
        The config module containing the parameters to check
    optional_vars :
        The optional parameters with predefined values
    key_var :
        The key to look at in optional_vars
    """
    for key , val in optional_vars[key_var].items():
        if not hasattr(cfg, key):
            setattr(cfg, key, val)
            logger.warning('Setting undefined parameter %s=%s' % (key, getattr(cfg, key)))

#-------------------------------------------------------------------------
def _check_cfg_mandatory(cfg, critical_vars, key_var):
    """    
    Check that the mandatory parameters are defined
    
    Parameters
    ----------
    cfg : python.module
        The config module containing the parameters to check
    critical_vars :
        The critival parameters needed for the protocol
    key_var :
        The key to look at in critical_vars
    """
    for v in critical_vars[key_var]:
        if not hasattr(cfg, v):
            logger.error('%s not defined in config.' % v)
            raise RuntimeError

#-------------------------------------------------------------------------
def _check_cfg_selected(cfg, optional_vars, select):
    """
    Used in case of dict attributes containing subparams
    Check that the selected cfg params is valid and that its
    subparameters are defined.

    Parameters
    ----------
    cfg : python.module
        The config module containing the parameters to check
    optional_vars :
        The optional parameters with predefined values for the param 
    selected = the cfg parameter (type=dict) containing a key: selected
    """
    param = getattr(cfg, select)
    selected = param['selected']
    
    if selected not in param:
        logger.error('%s not defined in config.'% selected)
        raise RuntimeError
    for v,vv in optional_vars[selected].items():
        if v not in param[selected]:
            param[selected].update({v: vv})
            setattr(cfg, select, param)
            logger.warning('Updating internal parameter for classifier %s: %s=%s' % (selected, v, vv))

#-------------------------------------------------------------------------
if __name__ == '__main__':
    # Load parameters
    if len(sys.argv) < 2:
        cfg_module = input('Config module name? ')
    else:
        cfg_module = sys.argv[1]
    batch_run(cfg_module)

    logger.info('Finished.')
