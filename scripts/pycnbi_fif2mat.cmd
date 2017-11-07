@ECHO OFF
IF NOT "%1" == "" (
    python -c "if __name__ == '__main__': from pycnbi.utils.fif2mat import fif2mat; fif2mat('%1')"
    pause
) ELSE (
    ECHO Usage: %0 [FIF_PATH]
)
