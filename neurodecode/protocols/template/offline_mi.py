import sys
import multiprocessing as mp

from neurodecode import logger
from neurodecode.utils.timer import Timer
from neurodecode.triggers import Trigger, TriggerDef
from neurodecode.gui.streams import redirect_stdout_to_queue

import neurodecode.utils.io as io

def check_config(cfg):
    '''
    Check the variables contained in the loaded config file

    Parameters
    ----------
    cfg : python.module
        The loaded config module
    '''

    # Add here the critical variables that need to be defined in the config_offline.py
    critical_vars = {
        'COMMON': ['TRIALS_NB',
                    'TRIGGER_FILE',
                    'TRIGGER_DEVICE'],
    }

    # Add here the optional variables that do not need to be defined in the config_offline.py
    # If not defined, the variable will be added with the value defined below
    optional_vars = {
        'COMMON' : {'REFRESH_RATE': 20, },

        # Internal parmameters for the CCC
        'XXX': { 'min': 1, 'max': 40, },
    }

    # Check the critical variables
    _check_cfg_mandatory(cfg, critical_vars, 'COMMON')

    # Check the optional variables
    _check_cfg_optional(cfg, optional_vars, 'COMMON')

    # Check the internal param of CCC
    _check_cfg_selected(cfg, optional_vars, 'CCC')

    # The TRIGGER_DEVICE attribute is mandatory
    if getattr(cfg, 'TRIGGER_DEVICE') == None:
        logger.warning('The trigger device is set to None! No events will be saved.')

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
    check_config(cfg)
    run(cfg)

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
    # Use to redirect sys.stdout to GUI terminal if GUI usage
    redirect_stdout_to_queue(logger, queue, 'INFO')

    # Wait the recording to start (GUI)
    while state.value == 2: # 0: stop, 1:start, 2:wait
        pass

    # Protocol start if equals to 1
    if not state.value:
        sys.exit()

    # Load the mapping from int to string for triggers events
    cfg.tdef = TriggerDef(cfg.TRIGGER_FILE)

    # Refresh rate
    refresh_delay = 1.0 / cfg.REFRESH_RATE

    # Trigger
    trigger = Trigger(lpttype=cfg.TRIGGER_DEVICE, state=state)
    if trigger.init(50) == False:
        logger.error('\n** Error connecting to the trigger device. Use a mock trigger instead?')
        input('Press Ctrl+C to stop or Enter to continue.')
        trigger = Trigger(lpttype='FAKE')
        trigger.init(50)

    # timers
    timer_refresh = Timer()

    trial = 1
    num_trials = cfg.TRIALS_NB

    # start
    while trial <= num_trials:
        timer_refresh.sleep_atleast(refresh_delay)
        timer_refresh.reset()

        #-------------------------------------
        # ADD YOUR CODE HERE
        #-------------------------------------

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
    if len(sys.argv) < 2:
        cfg_module = input('Config module name? ')
    else:
        cfg_module = sys.argv[1]
    batch_run(cfg_module)
