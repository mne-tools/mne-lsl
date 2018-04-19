@ECHO OFF
IF NOT "%1" == "" (
    python -c "if __name__ == '__main__': from pycnbi.utils.convert2fif import main; main('%1')"
) ELSE (
    ECHO Usage: %0 [FIF_PATH]
)
