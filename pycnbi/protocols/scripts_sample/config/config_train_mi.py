# Trigger device type ['ARDUINO', 'USB2LPT', 'SOFTWARE', 'DESKTOP', None]
TRIGGER_DEVICE = None
TRIGGER_DEF = 'triggerdef_16'  # full list: pycnbi.ROOT/Triggers/*.ini

# Screen property
SCREEN_SIZE = (1680, 1050)
SCREEN_POS = (1920, 0)

# Bar direction definition: L, R, U, D, B (both hands)
DIRECTIONS = ['L', 'R']

# Number of trials for each action
TRIALS_EACH = 20

# Timings
T_INIT = 10  # initial waiting time
T_GAP = 4  # show how many trials left
T_CUE = 2  # no bar, only red dot
T_DIR_READY = 2  # green bar
T_DIR = 5  # blue bar

# Use Google Glass?
GLASS_USE = False

# Maximum refresh rate in Hz
REFRESH_RATE = 30
