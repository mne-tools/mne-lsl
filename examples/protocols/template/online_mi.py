import sys
import multiprocessing as mp

import neurodecode.utils.io as io

from neurodecode import logger
from neurodecode.utils.timer import Timer
from neurodecode.triggers import Trigger, TriggerDef
from neurodecode.stream_receiver import StreamReceiver
from neurodecode.gui.streams import redirect_stdout_to_queue

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
        'COMMON': ['REFRESH_RATE',
                   'TRIGGER_FILE',
                   'TRIGGER_DEVICE'
                   'CCC'],
    }

    # Check the critical variables
    optional_vars = {
        'COMMON': { 'DDD',
                    'EEE'},

        # Internal parmameters for the CCC
        'XXX': { 'min': 1, 'max': 40, },
    }

    # Check the critical variables
    _check_cfg_mandatory(cfg, critical_vars, 'COMMON')

    # Check the optional variables
    _check_cfg_optional(cfg, optional_vars, 'COMMON')

    # Check the internal param of CCC
    _check_cfg_selected(cfg, optional_vars, 'CCC')

    return cfg

#-------------------------------------------------------------------------
def run(cfg, state=mp.Value('i', 1), queue=None):
    '''
    Main function used to run the online protocol.

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

    # Wait the recording to start (GUI)
    while state.value == 2: # 0: stop, 1:start, 2:wait
        pass

    #  Protocol runs if state equals to 1
    if not state.value:
        sys.exit(-1)

    # events and triggers
    cfg.tdef = TriggerDef(cfg.TRIGGER_FILE)

    # To send trigger events
    trigger = Trigger(cfg.TRIGGER_DEVICE, state)

    if trigger.init(50) == False:
        logger.error('Cannot connect to trigger device. Use a mock trigger instead?')
        input('Press Ctrl+C to stop or Enter to continue.')
        trigger = Trigger('FAKE', state)
        trigger.init(50)

    # Instance a stream receiver
    sr = StreamReceiver(bufsize=1, winsize=0.5,
                        stream_name=None, eeg_only=True)

    # Timer for acquisition rate, here 20 Hz
    tm = Timer(autoreset=True)

    # Refresh rate
    refresh_delay = 1.0 / cfg.REFRESH_RATE

    while True:

        # Acquire data from all the connected LSL streams by filling each associated buffers.
        sr.acquire()

        # Extract the latest window from the buffer of the chosen stream.
        window, tslist = sr.get_window()              # window = [samples x channels], tslist = [samples]

        #-------------------------------------
        # ADD YOUR CODE HERE
        #-------------------------------------

        #  To run a trained BCI decoder, look at online_mi.py protocol

        tm.sleep_atleast(refresh_delay)

    with state.get_lock():
        state.value = 0

    logger.info('Finished.')

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
