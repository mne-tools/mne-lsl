@ECHO OFF
IF NOT "%1" == "" (
    ipython -c "from pycnbi.utils.fif2mat import fif2mat; fif2mat('%1')"
) ELSE (
    ECHO Usage: %0 [FIF_PATH]
)
